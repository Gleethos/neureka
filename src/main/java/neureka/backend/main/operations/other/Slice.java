package neureka.backend.main.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.operations.ElemWiseUtil;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.devices.Device;
import neureka.framing.Relation;
import neureka.ndim.NDConstructor;
import org.slf4j.Logger;

import java.util.*;

public class Slice extends AbstractOperation
{
    private static final Logger _LOG = org.slf4j.LoggerFactory.getLogger( Slice.class );
    public Slice()
    {
        super(
            new OperationBuilder()
                .identifier(       "slice"     )
                .operator(         "slice"     )
                .arity(            1           )
                .isOperator(       false       )
                .isIndexer(        false       )
                .isDifferentiable( true        )
                .isInline(         false       )
        );
        setAlgorithm(
            Algorithm.withName("slice")
            .setIsSuitableFor( call -> SuitabilityPredicate.GOOD )
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
            .setExecution(
                ( caller, call ) ->
                {
                    int[] newShape    = call.getValOf(Arg.Shape.class);
                    int[] newOffset   = call.getValOf(Arg.Offset.class);
                    int[] newSpread   = call.getValOf(Arg.Stride.class);
                    Tsr<Object> input = (Tsr<Object>) call.input(0);
                    Tsr<?> subset     = _slice(newShape, newOffset, newSpread, input);
                    //---
                    Class<?>       typeClass = input.itemType();
                    int[]          shape = input.getNDConf().shape();
                    boolean        isOutsourced = input.isOutsourced();
                    Device<Object> device = input.getDevice();
                    //---
                    _sliceFrame( input, subset, newShape, newOffset, newSpread );
                    return
                        Result.of(subset.mut().setIsIntermediate(true))
                            .withADAction( t -> {
                                Tsr<Object> newError = ElemWiseUtil.newTsrLike((Class<Object>) typeClass, shape, isOutsourced, device, 0);
                                boolean isIntermediate = newError.isIntermediate();
                                newError.mut().setIsIntermediate(false); // To avoid deletion!
                                Tsr<Object> slice = Function.of("slice(I[0])", false)
                                                    .with(Arg.Shape.of(newShape),Arg.Offset.of(newOffset),Arg.Stride.of(newSpread))
                                                    .call(newError);

                                newError.mut().setIsIntermediate(isIntermediate);
                                slice.mut().setIsIntermediate(false);
                                Neureka.get().backend().getFunction().idy().execute( slice, t.error().mut().setIsVirtual(false) );
                                return newError;
                            });
                }
            )
            .buildFunAlgorithm()
        );
    }

    private static Tsr<?> _slice(
        int[] newShape,
        int[] newOffset,
        int[] newSpread,
        Tsr<Object> input
    ) {
        input.mut().setIsVirtual( false );
        int[] newTranslation = input.getNDConf().translation();
        int[] newIndicesMap = input.getNDConf().getLayout().newTranslationFor( newShape );

        for ( int i = 0; i < input.rank(); i++ )
            newSpread[ i ] = ( newSpread[i] == 0 ) ? 1 : newSpread[ i ];

        for ( int i = 0; i < newOffset.length; i++ )
            newOffset[ i ] = newOffset[ i ] + input.getNDConf().offset( i ); // Offset is being inherited!

        Relation<?> inputRelation = input.get( Relation.class );
        Tsr<?> rootTensor   = ( input.isSlice() ? inputRelation.findRootTensor().orElseThrow(IllegalStateException::new) : input );
        Tsr<?> parentTensor = ( input.isSlice() ? inputRelation.getParent().orElseThrow(IllegalStateException::new)      : input );
        /*
            The following code check the validity of the slice shape ranges with
            respect to the 'parentTensor' of this new slice.
         */
        if ( parentTensor.rank() == newShape.length && rootTensor == parentTensor ) {
            /*
                1. We know that inside this else branch 'this' tensor is a first order slice!
                (So it is not a slice of a slice... reason : 'rootTensor == parentTensor' )

                2. There is however uncertainty about the 'true shape' of this parent tensor!
                Meaning : It might have been permuted and could therefore be distorted with
                respect to the slice that is currently being prepared!
                -> This means we have to take this possible reshaping into account!
                Like so:

                The following uses an int array also called 'permuteRelation'.
                This is simply the 'permute array' which has been recorded inside the 'Relation' component
                by the 'Reshape' operation! ( Hopefully! :) ... custom shape operations need to consider this as well! )

                The following would occur when : "Tsr.of(...).T().getAt(...);"
                Transposing a tensor performs an inline reshaping of an identical
                slice of the original tensor! Then again slicing this tensor
                via the 'getAt(...)' method leads us to a situation where
                the following variable is NOT NULL! :
             */
            int[] permute = ( input.isSlice() ? parentTensor.get( Relation.class ).getPermuteRelationFor( input ) : null );
            permute = ( permute != null ) ? Permute.invert( permute ) : null;
            for ( int i = 0; i < parentTensor.rank(); i++ ) {
                int ii = ( permute != null ) ? permute[ i ] : i;
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
        else if ( rootTensor != parentTensor ) {
            // TODO! This requires some more thought about how handle slices of slices!
            _LOG.warn(
                "Exceptional higher order slice request detected. " +
                "This type of tensor cannot yet be sliced. " +
                "Please copy this tensor before slicing."
            );
        }

        Tsr<Object> subset =
                        Tsr.of(
                            input.getDataType(),
                            NDConstructor.of( newShape, newTranslation, newIndicesMap, newSpread, newOffset ),
                            input.mut().getData()
                        );

        subset.set( Relation.newChildToParent( input ) );
        Relation<Object> parent = input.find( Relation.class ).map(r->(Relation<Object>)r).orElseGet(Relation::newParentToChildren);
        parent.addChild( subset );
        input.set( parent );

        if ( input.isOutsourced() )
            input.getDevice().store( subset );

        if ( input.isVirtual() ) subset.mut().setIsVirtual( true );

        return subset;
    }

    private void _sliceFrame(
        Tsr<?> input, Tsr<?> subset, int[] newShape, int[] newOffset, int[] newSpread
    ) {
        // Now if the parent tensor has a name and or axes labels we carry them over to the subset:
        String label = input.label();
        if ( !label.isEmpty() ) subset.mut().label( label + ":slice" );
        input.frame().ifPresent( frame -> {
            Map<Object, List<Object>> state = frame.getState();
            Map<Object, List<Object>> sliceState = new LinkedHashMap<>();
            int i = 0;
            for ( Object k : state.keySet() ) {
                List<Object> axesLabels = state.get(k);
                if ( axesLabels == null )
                    sliceState.put( k, null ); // newShape[i]
                else {
                    List<Object> slicedLabels = new ArrayList<>();
                    for ( int j = 0; j < newShape[i]; j++ ) {
                        int index = newOffset[i] + j * newSpread[i];
                        slicedLabels.add( axesLabels.get(index) );
                    }
                    sliceState.put( k, slicedLabels );
                }
                i++;
                if ( i == newShape.length ) break;
            }
            subset.mut().labelAxes( sliceState );
        });

    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}

