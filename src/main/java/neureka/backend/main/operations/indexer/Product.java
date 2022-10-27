package neureka.backend.main.operations.indexer;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAction;
import neureka.backend.api.*;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.implementations.broadcast.CLBroadcastMultiplication;
import neureka.backend.main.implementations.broadcast.CPUBroadcastMultiplication;
import neureka.backend.main.operations.ElemWiseUtil;
import neureka.backend.main.operations.operator.Multiplication;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionParser;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.NDimensional;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 *  This type of operation belongs to the same species as the
 *  {@link Summation} operation.
 *  It executes incoming calls so that the calling function
 *  will be executed with all input indices passed to it.
 *  The resulting array of tensors will then multiplied with each other
 *  to produce the result of this operation, hence the name {@link Product}.
 */
public final class Product extends AbstractOperation
{
    public Product()
    {
        super (
            new OperationBuilder()
            .identifier(       "prodJs"    )
            .operator(         "prodJs"    )
            .arity(            1           )
            .isOperator(       false       )
            .isIndexer(        true        )
            .isDifferentiable( true        )
            .isInline(         false       )
        );

        setAlgorithm(
            new Broadcast(ElemWiseUtil::forMultiplications)
            .setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD )
            .setSupplyADActionFor(
                ( Function f, ExecutionCall<? extends Device<?>> call ) ->
                {
                    if ( call.autogradMode().allowsForward() )
                        throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                    Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                    Function mul = Neureka.get().backend().getFunction().mul();
                    if ( ctxDerivative != null ) {
                        return ADAction.of( target -> mul.execute( target.error(), ctxDerivative ) );
                    }
                    int d = call.getValOf( Arg.DerivIdx.class );
                    Tsr<?> derivative = f.executeDerive( call.inputs(), d );
                    return ADAction.of( target -> mul.execute( target.error(), derivative ) );
                }
            )
            .buildFunAlgorithm()
            .setImplementationFor( CPU.class, new CPUBroadcastMultiplication())
            .setImplementationFor( OpenCLDevice.class, new CLBroadcastMultiplication( this.getIdentifier() ) )
        );

    }

    @Override
    public Result execute( final Function caller, final ExecutionCall<?> call )
    {
        if ( call.getDerivativeIndex() >= 0 )
        {
            if ( !call.validate().allNotNullHaveSame(NDimensional::shape).isValid() )
                throw new IllegalArgumentException("The shapes of the operands of the multiplication operation must be equal! (when deriving nested functions)");

            Function noAD = Function.of( caller.toString(), false );
            Tsr<?>[] results = new Tsr[ call.arity() ];
            for ( int i = 0; i < results.length; i++ ) {
                ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flattenForIndexer( noAD, call.withArgs(Arg.VarIdx.of(i), Arg.DerivIdx.of(-1)) );
                results[ i ] = flatCall.input( 0 );
            }

            int d = call.getDerivativeIndex();
            int[] toBeDerived = IntStream.range(0,call.arity())
                                            .filter( i -> caller.dependsOn(d) )
                                            .toArray();

            Tsr<?>[] derivs = new Tsr[ call.arity() ];
            for ( int i = 0; i < results.length; i++ ) {
                int finalI = i;
                if ( Arrays.stream(toBeDerived).anyMatch(v -> v == finalI) ) {
                    ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flattenForIndexer(noAD, call.withArgs(Arg.VarIdx.of(i), Arg.DerivIdx.of(d)));
                    derivs[i] = flatCall.input(0);
                }
            }
            return Multiplication.derive( toBeDerived, results, i -> derivs[i] );
        }

        Tsr<?>[] inputs = new Tsr[ call.arity() ];
        for ( int i = 0; i < inputs.length; i++ ) {
            ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flattenForIndexer( caller, call.withArgs(Arg.VarIdx.of(i)) );
            inputs[ i ] = flatCall.input( 0 );
        }

        Operation mullOp = Neureka.get().backend().getOperation("*");
        Function mul = new FunctionParser(Neureka.get().backend())
                .parse( mullOp, inputs.length, caller.isDoingAD() );

        return mullOp.execute( mul, call.withInputs(inputs).withOperation(mullOp).withArgs(Arg.DerivIdx.of(-1)) );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src )
    {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double prod = 1;
            boolean nothingDone = true;
            for ( int Ii = 0; Ii < inputs.length; Ii++ ) {
                prod *= src[ 0 ].call( inputs, Ii );
                nothingDone = false;
            }
            if ( nothingDone ) return src[ 0 ].call( inputs, j );
            return prod;
        } else {
            double u, ud, v, vd;
            u = src[ 0 ].call( inputs, 0 );
            ud = src[ 0 ].derive(inputs, d, 0);
            for ( int ji = 1; ji < inputs.length; ji++ ) {
                v = src[ 0 ].call( inputs, ji );
                vd = src[ 0 ].derive( inputs, d, ji );
                ud = u * vd + v * ud;
                u *= v;
            }
            return ud;
        }
    }

    
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double prod = 1;
            boolean nothingDone = true;
            for ( int i = 0; i < inputs.length; i++ ) {
                prod *= src[ 0 ].call( inputs, i );
                nothingDone = false;
            }
            if ( nothingDone ) return src[ 0 ].call( inputs );
            return prod;
        } else {
            double u, ud, v, vd;
            u = src[ 0 ].call(inputs, 0);
            ud = src[ 0 ].derive(inputs, d, 0);
            for ( int j = 1; j < inputs.length; j++ ) {
                v = src[ 0 ].call( inputs, j );
                vd = src[ 0 ].derive( inputs, d, j );
                ud = u * vd + v * ud;
                u *= v;
            }
            return ud;
        }
    }


}
