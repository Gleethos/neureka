package neureka.backend.main.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.math.parsing.FunctionParser;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.framing.Relation;
import neureka.ndim.NDUtil;
import neureka.ndim.config.NDConfiguration;

public class Reshape extends AbstractOperation
{
    public Reshape()
    {
        super(
            new OperationBuilder()
                .identifier(       "reshape"  )
                .operator(         ","        )
                .arity(            -1         )
                .isOperator(       true       )
                .isIndexer(        false      )
                .isDifferentiable( true       )
                .isInline(         false      )
        );
        setAlgorithm(
            Algorithm
            .withName( "reshape" )
            .setIsSuitableFor( call -> SuitabilityPredicate.GOOD )
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
            .setExecution(
                ( caller, call ) ->
                {
                    Tsr<?>[] inputs = AbstractDeviceAlgorithm.flatten(caller, call).inputs();
                    int[] newForm = new int[ inputs.length - 1 ];
                    for ( int i = 0; i < inputs.length - 1; i++ )
                        newForm[ i ] = ( (Number) inputs[ i ].item( 0 ) ).intValue();

                    if ( call.getValOf( Arg.DerivIdx.class ) >= 0 ) //reverse reshape:
                        newForm = invert( newForm );

                    return Result.of(_reshaped( inputs[ inputs.length - 1 ], newForm, true ))
                            .withADAction( target -> new FunctionParser( Neureka.get().backend() ).parse( caller.toString(), false ).derive( new Tsr[]{ target.error() },0 ) );
                }
            )
            .buildFunAlgorithm()
        );
    }

    private static Tsr<?> _reshaped( Tsr<?> tensor, int[] newForm, boolean newTsr )
    {
        Tsr<?> parent = tensor;
        tensor = newTsr ? tensor.shallowCopy().mut().setIsIntermediate( true ) : tensor;
        NDConfiguration newNDC = tensor.getNDConf().newReshaped( newForm );
        _shapeCheck( newNDC.shape(), tensor );
        tensor.mut().setNDConf( newNDC );
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
                                    NDUtil.shapeString( newReshape ) + ":(I[ 0 ])",
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
    public double calculate( double[] inputs, int j, int d, Function[] src )
    {
        return src[ 0 ].call( inputs, j );
    }


    
    private static void _shapeCheck( int[] newShp, Tsr<?> t ) {
        if ( NDConfiguration.Utility.sizeOfShape( newShp ) != t.size() ) {
            throw new IllegalArgumentException(
                    "New shape does not match tensor size!" +
                    " (" +
                        NDUtil.shapeString( newShp ) +
                        ((NDConfiguration.Utility.sizeOfShape( newShp ) < t.size()) ? "<" : ">") +
                        NDUtil.shapeString(t.getNDConf().shape()) + "" +
                    ")"
                );
        }
    }
    
}
