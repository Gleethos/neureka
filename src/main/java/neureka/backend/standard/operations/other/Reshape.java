package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.fun.AutoDiffMode;
import neureka.backend.api.algorithms.fun.Result;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.calculus.internal.CalcUtil;
import neureka.devices.Device;
import neureka.framing.Relation;
import neureka.ndim.AbstractTensor;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;

public class Reshape extends AbstractOperation
{
    public Reshape()
    {
        super(
            new OperationBuilder()
                .setIdentifier(       "reshape"  )
                .setOperator(         ","        )
                .setArity(            -1         )
                .setIsOperator(       true       )
                .setIsIndexer(        false      )
                .setIsDifferentiable( true       )
                .setIsInline(         false      )
        );
        setAlgorithm(
            Algorithm
            .withName( "reshape" )
            .setIsSuitableFor( call -> SuitabilityPredicate.GOOD )
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
            .setExecution(
                ( caller, call ) ->
                {
                    Tsr<?>[] inputs = CalcUtil.srcActivation( call.inputs(), call.getValOf( Arg.VarIdx.class ), -1, 0, caller.getSubFunctions().toArray(new Function[0]) );
                    int[] newForm = new int[ inputs.length - 1 ];
                    for ( int i = 0; i < inputs.length - 1; i++ )
                        newForm[ i ] = ( (Number) inputs[ i ].getValueAt( 0 ) ).intValue();

                    if ( call.getValOf( Arg.DerivIdx.class ) >= 0 ) //reverse reshape:
                        newForm = invert( newForm );

                    return Result.of(_reshaped( inputs[ inputs.length - 1 ], newForm, true ))
                            .withAutoDiff( (Function f, ExecutionCall<? extends Device<?>> adCall, boolean forward ) ->
                            {
                                if ( forward )
                                    throw new IllegalArgumentException("Reshape operation does not support forward-AD!");

                                return ADAgent.of( null )
                                        .setForward( (t, derivative ) -> new FunctionBuilder( Neureka.get().backend() ).build( f.toString(), false ).derive( new Tsr[]{ derivative },0 ) )
                                        .setBackward( (t, error ) -> new FunctionBuilder( Neureka.get().backend() ).build( f.toString(), false ).derive( new Tsr[]{ error },0 ) );
                            });
                }
            )
            .setCallPreparation( call -> call )
            .buildFunAlgorithm()
        );
    }

    private static Tsr<?> _reshaped( Tsr<?> tensor, int[] newForm, boolean newTsr )
    {
        Tsr<?> parent = tensor;
        tensor = newTsr ? tensor.shallowCopy().getUnsafe().setIsIntermediate( true ) : tensor;
        NDConfiguration newNDC = tensor.getNDConf().newReshaped( newForm );
        _shapeCheck( newNDC.shape(), tensor );
        tensor.getUnsafe().setNDConf( newNDC );
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
        int[] endings = DimTrim.endsFrom( shape );
        int prefix = endings[0];
        int postfix = endings[1];
        for ( int i = 0; i < tensors.length; i++ ) {
            if ( tensors[ i ].rank() != largest ) {
                int[] oldShape = tensors[ i ].getNDConf().shape();
                int[] newReshape = new int[ largest ];
                int padding = largest - oldShape.length;

                int handle = ( postfix <= prefix ) ? padding : largest - padding;
                for ( int ii = 0; ii < handle; ii++ ) newReshape[ ii ] = ( postfix <= prefix ) ? -1 : ii;
                for ( int ii = handle; ii < largest; ii++ ) newReshape[ ii ] = ( postfix <= prefix ) ? ii - padding : -1;

                Function f = Function.of(
                        AbstractTensor.Utility.shapeString( newReshape ) + ":(I[ 0 ])",
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
                s -> {
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
            } else
                reconstructed.append(
                        ( isConstantNumeric.apply( children[ i ] ) )
                                ? children[ i ].split("\\.")[ 0 ]
                                : children[ i ]
                );

            if ( i < children.length - 2 )
                reconstructed.append(",");
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


    @Contract(pure = true)
    private static void _shapeCheck( int[] newShp, Tsr<?> t ) {
        if ( NDConfiguration.Utility.sizeOfShape( newShp ) != t.size() ) {
            throw new IllegalArgumentException(
                    "New shape does not match tensor size!" +
                            " (" +
                            AbstractTensor.Utility.shapeString( newShp ) +
                            ((NDConfiguration.Utility.sizeOfShape( newShp ) < t.size()) ? "<" : ">") +
                            AbstractTensor.Utility.shapeString(t.getNDConf().shape()) + "" +
                            ")"
            );
        }
    }
    
}
