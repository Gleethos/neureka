package neureka.backend.main.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAction;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.BiElementWise;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.memory.MemUtil;
import neureka.backend.main.operations.ElemWiseUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionParser;
import neureka.devices.Device;
import neureka.ndim.NDimensional;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class Multiplication extends AbstractOperation
{
    public Multiplication()
    {
        super(
                new OperationBuilder()
                        .identifier(    "multiply"    )
                        .operator(         "*"        )
                        .arity(            -1         )
                        .isOperator(       true       )
                        .isIndexer(        false      )
                        .isDifferentiable( true       )
                        .isInline(         false      )
        );

        setAlgorithm(
            BiElementWise.class,
            new BiElementWise(ElemWiseUtil::forMultiplications)
            .setSupplyADActionFor( getDefaultAlgorithm() )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            Broadcast.class,
            new Broadcast( ElemWiseUtil::forMultiplications )
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
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
                    int d = call.getDerivativeIndex();
                    Tsr<?> derivative = MemUtil.keep( call.inputs(), () -> f.executeDerive( call.inputs(), d ) );
                    return ADAction.of( target -> mul.execute( target.error(), derivative ) );
                }
            )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            Scalarization.class,
            new Scalarization()
            .setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD )
            .setDeviceExecution( (call, callback) -> ElemWiseUtil.forMultiplications(call, callback) )
            .buildFunAlgorithm()
        );
    }

    @Override
    public Result execute(Function caller, ExecutionCall<?> call )
    {
        if ( !caller.isFlat() ) {
            int d = call.getDerivativeIndex();
            if ( d < 0 ) {
                ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flatten( caller, call.withArgs(Arg.DerivIdx.of(-1)) );
                Function flat = new FunctionParser(Neureka.get().backend()).parse( flatCall.getOperation(), flatCall.arity(), true );
                Result r = super.execute( flat, flatCall );
                //for ( int i = 0; i < flatCall.inputs().length; i++ )
                //    _deleteIfNotIn(call.inputs(), flatCall.input(i)); // TODO: Make it possible to delete more stuff
                return r;
            } else {
                if ( !call.validate().allNotNullHaveSame(NDimensional::shape).isValid() )
                    throw new IllegalArgumentException("The shapes of the operands of the multiplication operation must be equal! (when deriving nested functions)");

                Function noAd = Function.of( caller.toString(), false );
                ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flatten( noAd, call.withArgs(Arg.DerivIdx.of(-1)) );

                int[] toBeDerived = IntStream.range(0,caller.getSubFunctions().size())
                        .filter( i -> caller.getSubFunctions().get(i).dependsOn(d) )
                        .toArray();

                Tsr[] derivatives = new Tsr[ toBeDerived.length ];
                Tsr[] results = flatCall.inputs();
                Function mul = Neureka.get().backend().getFunction().mul();
                Function add = Neureka.get().backend().getFunction().add();
                Tsr<?> finalDerivative = null;
                for ( int i = 0; i < derivatives.length; i++ ) {
                    Function noAD = Function.of( caller.getSubFunctions().get( toBeDerived[i] ).toString(), false );
                    Tsr<?> deriv = noAD.execute( noAD.getOperation() == null ? call : call.withOperation(noAD.getOperation()) );
                    derivatives[ i ] = deriv;
                    Tsr<?> localDeriv = null;
                    for ( int j = 0; j < results.length; j++ ) {
                        // Now we calculate the local derivatives of the multiplication operation:
                        if ( j == toBeDerived[i] ) {
                            if ( localDeriv == null ) localDeriv = derivatives[ i ];
                            else localDeriv = mul.execute( localDeriv, derivatives[ i ] );
                        } else {
                            if ( localDeriv == null ) localDeriv = results[ j ];
                            else localDeriv = mul.execute( localDeriv, results[ j ] );
                        }
                    }
                    if ( finalDerivative == null ) finalDerivative = localDeriv;
                    else finalDerivative = add.execute( finalDerivative, localDeriv );
                }
                return Result.of( finalDerivative );
            }
        }
        return super.execute( caller, call );
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        return Arrays.stream( children )
                .filter( child -> child.dependsOn(derivationIndex) )
                .map( child -> {
                            String derivative = child.getDerivative(derivationIndex).toString();
                            return ( (derivative.equals("1.0") ) ? "" : " * " ) +
                                    Arrays.stream( children )
                                        .filter( inner -> inner != child )
                                        .map( Object::toString )
                                        .collect( Collectors.joining( " * " ) );
                        }
                )
                .map( Object::toString )
                .collect( Collectors.joining( " + " ) );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result *= current;
            }
            return result;
        } else {
            double u, ud, v, vd;
            u = src[ 0 ].call( inputs, j );
            ud = src[ 0 ].derive( inputs, d, j );

            for ( int ji = 1; ji < src.length; ji++ ) {
                v = src[ ji ].call( inputs, j );
                vd = src[ ji ].derive( inputs, d, j );
                ud = u * vd + v * ud;
                u *= v;
            }
            return ud;
        }
    }

    
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result *= current;
            }
            return result;
        } else {
            double u, ud, v, vd;
            u = src[ 0 ].call( inputs );
            ud = src[ 0 ].derive( inputs, d );
            for ( int j = 1; j < src.length; j++ ) {
                v = src[ j ].call( inputs );
                vd = src[ j ].derive( inputs, d );

                ud = u * vd + v * ud;
                u *= v; // ...this step can be avoided (TODO optimize)
            }
            return ud;
        }
    }




}
