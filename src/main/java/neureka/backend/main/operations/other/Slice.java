package neureka.backend.main.operations.other;

import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.framing.Relation;
import neureka.ndim.NDConstructor;
import org.slf4j.Logger;

public class Slice extends AbstractOperation
{
    private static final Logger _LOG = org.slf4j.LoggerFactory.getLogger( Slice.class );
    public Slice()
    {
        super(
            new OperationBuilder()
                .setIdentifier(       "slice"    )
                .setOperator(         "slice"    )
                .setArity(            1           )
                .setIsOperator(       false       )
                .setIsIndexer(        false       )
                .setIsDifferentiable( true        )
                .setIsInline(         false       )
        );
        setAlgorithm(
            Algorithm.withName("slice")
            .setIsSuitableFor( call -> SuitabilityPredicate.GOOD )
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
            .setExecution(
                ( caller, call ) ->
                {
                    int[] newShape  = call.getValOf(Arg.Shape.class);
                    int[] newOffset = call.getValOf(Arg.Offset.class);
                    int[] newSpread = call.getValOf(Arg.Stride.class);
                    Tsr<Object> target = (Tsr<Object>) call.input(0);
                    target.setIsVirtual( false );
                    int[] newTranslation = target.getNDConf().translation();
                    int[] newIndicesMap = target.getNDConf().getLayout().newTranslationFor( newShape );

                    for ( int i = 0; i < target.rank(); i++ )
                        newSpread[ i ] = ( newSpread[i] == 0 ) ? 1 : newSpread[ i ];

                    for ( int i = 0; i < newOffset.length; i++ )
                        newOffset[ i ] = newOffset[ i ] + target.getNDConf().offset( i ); // Offset is being inherited!

                    Tsr<?> rootTensor   = ( target.isSlice() ? target.get( Relation.class ).findRootTensor() : target );
                    Tsr<?> parentTensor = ( target.isSlice() ? target.get( Relation.class ).getParent()      : target );
                    /*
                        The following code check the validity of the slice shape ranges with
                        respect to the 'parentTensor' of this new slice.
                     */
                    if ( parentTensor.rank() != newShape.length || rootTensor != parentTensor ) {
                        // TODO! This requires some more thought about how to check this!
                        // THIS CASE HAS NOT YET BEEN THOUGHT TROUGH!
                        _LOG.warn(
                                "Exceptional slice request detected. " +
                                        "This type of tensor cannot yet be sliced. " +
                                        "Please copy this tensor before slicing."
                        );
                    } else {
                        /*
                            1. We know that inside this else branch 'this' tensor is a first order slice!
                            (So it is not a slice of a slice... reason : 'rootTensor == parentTensor' )

                            2. There is however uncertainty about the 'true shape' of this parent tensor!
                            Meaning : It might have been reshaped and could therefore be distorted with
                            respect to the slice that is currently being prepared!
                            -> This means we have to take this possible reshaping into account!
                            Like so:

                            The following uses an int array also called 'reshapeRelation'.
                            This is simply the 'reshape array' which has been recorded inside the 'Relation' component
                            by the 'Reshape' operation! ( Hopefully! :) ... custom shape operations need to consider this as well! )

                            The following would occur when : "Tsr.of(...).T().getAt(...);"
                            Transposing a tensor performs an inline reshaping of an identical
                            slice of the original tensor! Then again slicing this tensor
                            via the 'getAt(...)' method leads us to a situation where
                            the following variable is NOT NULL! :
                         */
                        int[] reshaped = ( target.isSlice() ) ? parentTensor.get( Relation.class ).getReshapeRelationFor( target ) : null;
                        reshaped = ( reshaped != null ) ? Reshape.invert( reshaped ) : null;
                        for ( int i = 0; i < parentTensor.rank(); i++ ) {
                            int ii = ( reshaped != null ) ? reshaped[ i ] : i;
                            int top = newOffset[ i ] + newShape[ i ];
                            if ( top > parentTensor.shape( ii ) ) {
                                String message =
                                        "Cannot create slice because ranges are out of the bounds of the targeted tensor.\n" +
                                                "At index '" + i + "' : offset '" + newOffset[ i ] + "' + shape '" + newShape[ i ] + "' = '" + top + "',\n" +
                                                "which is larger than the target shape '" + parentTensor.shape( ii ) + "' at the same index!";
                                Exception exception = new IllegalArgumentException( message );
                                _LOG.error( message, exception );
                                throw new IllegalArgumentException( exception );
                            }
                        }
                    }

                    Tsr<Object> subset =
                            Tsr.of(
                                    target.getDataType(),
                                    NDConstructor.of( newShape, newTranslation, newIndicesMap, newSpread, newOffset ),
                                    target.getUnsafe().getData()
                            );

                    if ( target.isOutsourced() ) {
                        Device<Object> device = target.getDevice();
                        device.store( subset, target );
                        subset.setIsOutsourced( true );
                    }
                    if ( target.isVirtual() ) subset.setIsVirtual( true );
                    subset.set( new Relation().addParent( target ) );
                    Relation<Object> parent = target.get( Relation.class );
                    parent = ( parent != null ) ? parent : new Relation<>();
                    parent.addChild( subset );
                    target.set( parent );
                    //return subset;

                    return
                        Result.of(subset.getUnsafe().setIsIntermediate(true));
                            //.withADAction( t -> {
                            //  return null; // TODO
                            //});
                }
            )
            .buildFunAlgorithm()
        );
    }

    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.charAt( 0 ) == '(' && expression.charAt( expression.length() - 1 ) == ')' ) {
            return "slice" + expression;
        }
        return "slice" + "(" + expression + ")";
    }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}

