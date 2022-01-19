package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.FunAlgorithm;
import neureka.calculus.internal.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.Device;
import neureka.framing.Relation;
import neureka.ndim.AbstractNDArray;
import neureka.ndim.config.NDConfiguration;

import java.util.ArrayList;

public class Reshape extends AbstractOperation
{

    public Reshape()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "reshape"    )
                        .setOperator(         ","        )
                        .setArity(            -1         )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( true       )
                        .setIsInline(         false      )
        );

        FunAlgorithm implementation =
                Algorithm.withName( "reshape" )
                            .setIsSuitableFor( call -> SuitabilityPredicate.GOOD )
                            .setCanPerformBackwardADFor( call -> true )
                            .setCanPerformForwardADFor( call -> false )
                            .setSupplyADAgentFor(
                                ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                {
                                    //Tsr ctxDerivative = (Tsr)call.findAndGet(Argument.Derivative.class);
                                    if ( forward ) {
                                        throw new IllegalArgumentException("Reshape operation does not support forward-AD!");
                                    }
                                    return ADAgent.of( null )
                                                    .setForward( (t, derivative ) -> new FunctionBuilder( Neureka.get().backend() ).build( f.toString(), false ).derive( new Tsr[]{ derivative },0 ) )
                                                    .setBackward( (t, error ) -> new FunctionBuilder( Neureka.get().backend() ).build( f.toString(), false ).derive( new Tsr[]{ error },0 ) );
                                }
                            )
                            .setExecutionDispatcher(
                                ( caller, call ) ->
                                {
                                    Tsr<?>[] inputs = CalcUtil.srcActivation( call.getTensors(), call.getJ(), -1, 0, caller.getSubFunctions().toArray(new Function[0]) );
                                    int[] newForm = new int[ inputs.length - 1 ];
                                    for ( int i = 0; i < inputs.length - 1; i++ ) {
                                        newForm[ i ] = (int) Tsr.IO.getFrom( inputs[ i ], 0 );
                                    }
                                    if ( call.getValOf( Arg.DerivIdx.class ) >= 0 ) {//reverse reshape:
                                        newForm = invert( newForm );
                                    }
                                    Tsr<?> t = inputs[ inputs.length - 1 ];
                                    return reshaped( t, newForm, true );
                                }
                            )
                            .setCallPreparation( call -> call)
                            .buildFunAlgorithm();

        setAlgorithm(
                FunAlgorithm.class,
                implementation
        );

    }

    public static Tsr<?> reshaped( Tsr<?> tensor, int[] newForm, boolean newTsr )
    {
        Tsr<?> parent = tensor;
        tensor = (newTsr) ? tensor.getAt( new ArrayList<>() ).getMutate().setIsIntermediate( true ) : tensor;
        NDConfiguration newNDC = tensor.getNDConf().newReshaped( newForm );
        AbstractNDArray.Utility.Indexing.shpCheck( newNDC.shape(), tensor );
        tensor.getMutate().setNDConf( newNDC );
        if ( newTsr ) {
            Relation r = parent.get( Relation.class );
            r.addReshapeRelationFor( tensor, newForm );
        }
        return tensor;
    }


    public static void makeFit( Tsr<?>[] tensors, boolean doesAD )
    {
        int largest = -1;
        int[] shape = null;
        for ( Tsr<?> t : tensors ) if ( t.rank() > largest ) {
            largest = t.rank();
            shape = t.getNDConf().shape();
        }
        int prefix = 0;
        assert shape != null;
        for ( int s : shape ) if ( s == 1 ) prefix++; else break;
        int postfix = 0;
        for ( int i = shape.length-1; i>=0; i-- ) if ( shape[ i ] == 1 ) postfix++; else break;
        for ( int i = 0; i < tensors.length; i++ ) {
            if ( tensors[ i ].rank() != largest ) {
                int[] oldShape = tensors[ i ].getNDConf().shape();
                int[] newReshape = new int[ largest ];
                int padding = largest - oldShape.length;

                int handle = ( postfix <= prefix ) ? padding : largest - padding;
                for ( int ii = 0; ii < handle; ii++ ) newReshape[ ii ] = ( postfix <= prefix ) ? -1 : ii;
                for ( int ii = handle; ii < largest; ii++ ) newReshape[ ii ] = ( postfix <= prefix ) ? ii - padding : -1;

                Function f = Function.of(
                        AbstractNDArray.Utility.Stringify.strConf( newReshape ) + ":(I[ 0 ])",
                        doesAD
                );
                tensors[ i ] = f.execute( tensors[ i ] );
            }
        }

    }


    public static int[] invert( int[] reshape )
    {
        int reverseLength = 0;
        for ( int e : reshape ) {
            if ( e >= 0 ) reverseLength++;
        }
        int[] reversed = new int[ reverseLength ];
        int reshape_i = 0;
        int reverse_i = 0;
        while ( reverse_i < reverseLength ) {
            if ( reshape[ reshape_i ] >= 0 ) {
                reversed[ reshape[ reshape_i ] ] = reshape_i;
                reverse_i++;
            }
            reshape_i++;
        }
        return reversed;
    }

    @Override
    public String stringify( String[] children ) {
        java.util.function.Function<String, Boolean> isConstantNumeric =
                s ->
                {
                    try {
                        Double.parseDouble(s);
                        return true;
                    } catch (Exception e) { return false; }
                };
        StringBuilder reconstructed = new StringBuilder();
        reconstructed.insert(0, "[");
        for ( int i = 0; i < children.length; ++i ) {
            if ( i == children.length - 1 ) {
                reconstructed.append("]:(").append(
                        ( isConstantNumeric.apply( children[ i ] ) )
                                ? children[ i ].split("\\.")[ 0 ]
                                : children[ i ]
                ).append(")");
            } else {
                reconstructed.append(
                        ( isConstantNumeric.apply( children[ i ] ) )
                                ? children[ i ].split("\\.")[ 0 ]
                                : children[ i ]
                );
            }
            if ( i < children.length - 2 ) {
                reconstructed.append(",");
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src )
    {
            return src[ 0 ].call( inputs, j );
    }
}
