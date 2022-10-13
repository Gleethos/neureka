package neureka.backend.main.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAction;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.BiElementWise;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.internal.RecursiveExecutor;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Power extends AbstractOperation
{
    public Power()
    {
        super(
            new OperationBuilder()
            .identifier(       "power"    )
            .operator(         "**"        )
            .arity(            -1         )
            .isOperator(       true       )
            .isIndexer(        false      )
            .isDifferentiable( true       )
            .isInline(         false      )
        );

        //_____________________
        // DEFAULT OPERATION :

        RecursiveExecutor rja = (call, traverse)->
        {
            call = call.withInputs(call.inputs().clone()); // Let's make a copy of the inputs to avoid side effects.
            Device<Number> device = call.getDeviceFor(Number.class);
            int d = call.getValOf( Arg.DerivIdx.class );
            Operation type = call.getOperation();

            Tsr<?> result = null;
            if ( call.arity() > 3 )
            {
                if ( d < 0 ) {
                    Tsr<?>[] reduction = new Tsr[]{call.input( 0 ), call.input( 1 ), call.input( 2 )};
                    call = call.withInputAt( 0, traverse.execute( call.withInputs( reduction ) ) );
                    reduction = Utility.offsetted( call.inputs(), 1 );
                    result = traverse.execute( call.withInputs( reduction ) );
                    call = call.withInputAt( 0, result );
                } else {

                    Tsr<?>[] reduction = Utility.subset(call.inputs(), 1,  2, call.arity()-2);
                    reduction[ 0 ] = call.input( 1 ).deepCopy().mut().setIsIntermediate( true );

                    if ( d==0 ) {
                        Tsr<?> exp = traverse.execute(
                                                ExecutionCall.of( reduction )
                                                            .andArgs(Arg.DerivIdx.of( -1 ))
                                                            .running(Neureka.get().backend().getOperation("*"))
                                                            .on(device)
                                            );

                        reduction = new Tsr[]{call.input( 0 ), call.input( 1 ), exp};
                        call = call.withInputAt( 0, traverse.execute(
                                                    ExecutionCall.of( reduction )
                                                                    .andArgs( Arg.DerivIdx.of(0) )
                                                                    .running(type)
                                                                    .on( device )
                                                ));
                        exp.mut().delete();
                    } else {
                        Tsr<?> inner = traverse.execute(
                                                ExecutionCall.of( reduction )
                                                                .andArgs(Arg.DerivIdx.of(d-1))
                                                                .running(Neureka.get().backend().getOperation("*"))
                                                                .on(device)
                                        );

                        reduction = new Tsr[]{ call.input( 1 ).deepCopy().mut().setIsIntermediate( true ), inner, call.input( d ) };
                        Tsr<?> exp = traverse.execute(
                                                ExecutionCall.of( reduction )
                                                                .andArgs( Arg.DerivIdx.of(-1) )
                                                                .running( Neureka.get().backend().getOperation("*") )
                                                                .on( device )
                                      );

                        reduction = new Tsr[]{call.input( 0 ), call.input( 1 ), exp};
                        result = traverse.execute(
                                                ExecutionCall.of( reduction )
                                                            .andArgs(Arg.DerivIdx.of(1))
                                                            .running(type)
                                                            .on(device)
                                            );

                        call = call.withInputAt( 0, result );

                        inner.mut().delete();
                        exp.mut().delete();
                    }
                }
            }
            if ( result == null ) return AbstractDeviceAlgorithm.executeDeviceAlgorithm( call, null );
            return result;
        };

        setAlgorithm(BiElementWise.class,
            new BiElementWise( rja )
            .setSupplyADActionFor( getDefaultAlgorithm() )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            Broadcast.class,
            new Broadcast(rja)
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
                    int d = call.getDerivativeIndex();
                    Tsr<?> derivative = f.executeDerive( call.inputs(), d );
                    return ADAction.of( target -> mul.execute( target.error(), derivative ) );
                }
            )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            Scalarization.class,
            new Scalarization()
            .setIsSuitableFor( call -> SuitabilityPredicate.BAD )
            .setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD )
            .setDeviceExecution( (call, callback) -> rja.execute(call, callback) )
            .buildFunAlgorithm()
        );

    }

    @Override
    public Result execute( final Function caller, final ExecutionCall<?> call )
    {
        return super.execute( reducePairwise(caller), call );
    }

    private Function reducePairwise( final Function fun ) {
        Function reduced = fun;
        if ( reduced.getSubFunctions().size() > 2 ) {
            /*
                So currently we have something like this: a**b**c**d...
                However, this is how it is really executed:  (a**(b**(c**(d**..))))
                ...so let's create a function that is nested like the above:
            */
            Function nested = reduced.getSubFunctions().get(reduced.getSubFunctions().size()-1);
            for ( int i = reduced.getSubFunctions().size()-2; i >= 0; i-- )
                nested = Function.of( reduced.getSubFunctions().get(i) + "**" + nested, true );

            reduced = nested;
        }
        return reduced;
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        Function a = children[0];
        Function b = Function.of(
                IntStream.range( 1, children.length )
                .mapToObj(i -> children[ i ].toString() )
                .collect(Collectors.joining(" * "))
        );
        boolean aDerivable = a.dependsOn(derivationIndex);
        boolean bDerivable = b.dependsOn(derivationIndex);
        String aAsStr = a.toString();
        String bAsStr = b.toString();
        String first = "";
        if (aDerivable) {
            String aAsDerivative = a.getDerivative(derivationIndex).toString();
            if ( !aAsDerivative.equals("0.0") ) {
                first = ("( "+ bAsStr +" * "+ aAsStr + " ** (" + bAsStr + " - 1) )");
                if (!aAsDerivative.equals("1.0")) first = aAsDerivative + " * " + first;
            }
        }
        String bAsDerivative = "";
        if (bDerivable) bAsDerivative = b.getDerivative(derivationIndex).toString();
        if ( !bAsDerivative.isEmpty() && !bAsDerivative.equals("1.0") ) bAsDerivative += " * ";
        else bAsDerivative = "";
        String second = "";
        if ( bDerivable ) second = "(ln("+aAsStr+") * "+aAsStr+" ** "+bAsStr+")";
        String result;
        if ( !first.trim().isEmpty() && !second.trim().isEmpty() ) result = bAsDerivative+"("+first+" + "+second+")";
        else if (!first.trim().isEmpty()) result = bAsDerivative + "("+first+")";
        else if (!second.trim().isEmpty()) result = bAsDerivative + "(" +second + ")";
        else result = bAsDerivative;
        return result;
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result = Math.pow(result, current);
            }
            return result;
        } else {
            double b = 1;
            double bd = 0;
            double a;
            for ( int i = 1; i < src.length; i++ ) {
                double dd = 1;
                a = src[ i ].call( inputs, j );
                for ( int di = 1; di < src.length; di++ ) {
                    if ( di != i ) dd *= a;
                    else dd *= src[ di ].derive( inputs, d, j );
                }
                bd += dd;
                b *= a;
            }
            double out = 0;
            a = src[ 0 ].call( inputs, j );
            out += src[ 0 ].derive( inputs, d, j ) * b * Math.pow(a, b - 1);
            out += (a >= 0) ? bd *  Math.pow(a, b) * Math.log(a) : 0;
            return out;
        }
    }
    
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result = Math.pow(result, current);
            }
            return result;
        } else {
            double b = 1;
            double bd = 0;
            double a;
            for ( int i = 1; i < src.length; i++ ) {
                double dd = 1;
                a = src[ i ].call( inputs );
                for ( int di = 1; di < src.length; di++ ) {
                    if ( di != i ) dd *= a;
                    else dd *= src[ di ].derive( inputs, d );
                }
                bd += dd;
                b *= a;
            }
            double out = 0;
            a = src[ 0 ].call( inputs );
            out += src[ 0 ].derive( inputs, d ) * b * Math.pow(a, b - 1);
            out += (a >= 0) ? bd *  Math.pow(a, b) * Math.log(a) : 0;
            return out;
        }
    }

}
