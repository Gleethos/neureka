package neureka.backend.main.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAction;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.BiElementWise;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.operations.ElemWiseUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionParser;
import neureka.devices.Device;
import neureka.ndim.NDimensional;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Subtraction extends AbstractOperation
{
    public Subtraction()
    {
        super(
            new OperationBuilder()
            .identifier(    "subtract"    )
            .operator(         "-"        )
            .arity(            -1         )
            .isOperator(       true       )
            .isIndexer(        false      )
            .isDifferentiable( true       )
            .isInline(         false      )
        );

        setAlgorithm(
            new BiElementWise(ElemWiseUtil::forSubtractions)
            .setSupplyADActionFor( getDefaultAlgorithm() )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            Scalarization.class,
            new Scalarization()
            .setIsSuitableFor( call -> SuitabilityPredicate.BAD )
            .setDeviceExecution( (call, callback) -> ElemWiseUtil.forSubtractions(call, callback) )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            Broadcast.class,
            new Broadcast(ElemWiseUtil::forSubtractions)
                .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
                .setSupplyADActionFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call ) ->
                    {
                        if ( call.autogradMode().allowsForward() )
                            throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                        Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                        assert ctxDerivative == null;
                        int d = call.getDerivativeIndex();
                        Tsr<?> derivative = ElemWiseUtil.newTsrLike( call.input( d==0?1:0 ), 0 );
                        Tsr<?> toBeDerived = ElemWiseUtil.newTsrLike( call.input( d ), 0 );
                        Device device = call.getDevice();
                        return
                            ADAction.of(
                                target ->
                                    this.getAlgorithm( Broadcast.class )
                                        .getImplementationFor( device )
                                        .run(
                                            ExecutionCall.of(
                                                    toBeDerived.setIsVirtual(false),
                                                    derivative,
                                                    target.error()
                                                )
                                                .andArgs( Arg.DerivIdx.of(d) )
                                                .running( this )
                                                .on( device )
                                        )
                            );
                    }
                )
                .buildFunAlgorithm()
            );
    }

    @Override
    public Result execute(Function caller, ExecutionCall<?> call )
    {
        if ( !caller.isFlat() ) {
            int d = call.getDerivativeIndex();
            if ( d < 0 ) {
                caller = reducePairwise(caller);
                ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flatten( caller, call.withArgs(Arg.DerivIdx.of(-1)) );
                Function flat = new FunctionParser(Neureka.get().backend()).parse( flatCall.getOperation(), flatCall.arity(), true );
                return super.execute( flat, flatCall );
            } else {
                if ( !call.validate().allNotNullHaveSame(NDimensional::shape).isValid() )
                    throw new IllegalArgumentException("The shapes of the operands of the subtraction operation must be equal! (when deriving nested functions)");

                Function finalCaller = caller;
                int[] toBeDerived = IntStream.range(0,caller.getSubFunctions().size())
                                                        .filter( i -> finalCaller.getSubFunctions().get(i).dependsOn(d) )
                                                        .toArray();

                Tsr[] results = new Tsr[ toBeDerived.length ];
                Function neg = Neureka.get().backend().getFunction().neg();
                for ( int i = 0; i < results.length; i++ ) {
                    Function noAD = Function.of( caller.getSubFunctions().get( toBeDerived[i] ).toString(), false );
                    Tsr<?> deriv = noAD.execute( noAD.getOperation() == null ? call : call.withOperation(noAD.getOperation()) );
                    if ( i > 0 ) deriv = neg.execute(deriv);
                    results[ i ] = deriv;
                }
                if ( results.length == 1 ) return Result.of( results[0] );
                Function addAll = new FunctionParser(Neureka.get().backend()).parse(Neureka.get().backend().getOperation("+"), results.length, false);
                return addAll.getOperation().execute(addAll, call.withOperation(addAll.getOperation()).withInputs(results).withArgs(Arg.DerivIdx.of(-1)));
            }
        }
        caller = reducePairwise(caller);
        return super.execute( caller, call );
    }

    private Function reducePairwise(Function f) {
        if ( f.getSubFunctions().size() > 2 ) {
            /*
                So currently we have something like this: a-b-c-d...
                However, this is how it is really executed:  ((((a-b)-c)-d)..)
                ...so let's create a function that is nested like the above:
            */
            Function nested = f.getSubFunctions().get(0);
            for ( int i = 1; i < f.getSubFunctions().size(); i++ )
                nested = Function.of( nested + " - " + f.getSubFunctions().get(i), true );

            f = nested;
        }
        return f;
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        return ( ( children[0].dependsOn(derivationIndex) ) ? "" : "-" ) +
                    Arrays.stream( children )
                    .filter( child -> child.dependsOn(derivationIndex) )
                    .map( child -> child.getDerivative(derivationIndex) )
                    .map( Object::toString )
                    .collect( Collectors.joining( " - " ) );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result -= current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( int i = 0; i < src.length; i++ ) {
                if ( i == 0 )
                    derivative += src[ i ].derive( inputs, d, j );
                else
                    derivative -= src[ i ].derive( inputs, d, j );
            }
            return derivative;
        }
    }

    
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result -= current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( int i = 0; i < src.length; i++ ) {
                if ( i == 0 )
                    derivative += src[ i ].derive( inputs, d );
                else
                    derivative -= src[ i ].derive( inputs, d );
            }
            return derivative;
        }
    }



}
