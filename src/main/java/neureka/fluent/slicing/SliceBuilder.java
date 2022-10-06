package neureka.fluent.slicing;

import neureka.Tsr;
import neureka.calculus.args.Arg;
import neureka.fluent.slicing.states.AxisOrGetTsr;
import neureka.fluent.slicing.states.FromOrAt;
import neureka.fluent.slicing.states.FromOrAtTsr;

import java.util.function.Function;


/**
 *  This class is the heart of the slice builder API, collecting range configurations by
 *  exposing an API consisting of multiple interfaces which form a call state transition graph.
 *  Instances of this class do not perform the actual slicing of a {@link Tsr} instance themselves,
 *  however instead they merely serve as collectors of slice configuration data.
 *  The API exposed by the {@link SliceBuilder} uses method chaining as well as a set of implemented interfaces
 *  which reference themselves in the form of the return types defined by the method signatures of said interfaces.
 *  A user of the API can only call methods exposed by the current "view" of the builder, namely a interface.
 *  This ensures a controlled order of calls to the API...
 *
 * @param <V> The type of the value(s) held by the tensor which ought to be sliced with the help of this builder.
 */
public class SliceBuilder<V> implements AxisOrGetTsr<V>
{
    private interface CreationCallback<V> {
        Tsr<V> sliceOf(int[] newShape, int[] newOffset, int[] newSpread, boolean autograd);
    }

    private final Function<Boolean, Tsr<V>> _create;
    private final AxisSliceBuilder<V>[]  _axisSliceBuilders;

    /**
     *  An instance of a slice builder does not perform the actual slicing itself!
     *  Instead, it merely serves as a collector of slice configuration data.
     *  The actual slicing will be performed by the {@link CreationCallback} passed
     *  to this constructor.
     *
     * @param toBeSliced The {@link Tsr} instance which ought to be sliced.
     */
    public SliceBuilder( Tsr<V> toBeSliced )
    {
        CreationCallback<V> sliceCreator = // A callback lambda which receives the final slice configuration to perform the actual slicing.
        ( int[] newShape, int[] newOffset, int[] newSpread, boolean allowAutograd )->
        {
            boolean isIntermediate = toBeSliced.isIntermediate();
            toBeSliced.getMut().setIsIntermediate(false); // To avoid deletion!
            Tsr<V> slice = neureka.calculus.Function.of("slice(I[0])", allowAutograd)
                                .with(Arg.Shape.of(newShape),Arg.Offset.of(newOffset),Arg.Stride.of(newSpread))
                                .call(toBeSliced);

            slice.getMut().setIsIntermediate(false);
            toBeSliced.getMut().setIsIntermediate(isIntermediate);
            return slice;
        };

        int[] shape = toBeSliced.getNDConf().shape();
        _axisSliceBuilders = new AxisSliceBuilder[ shape.length ];
        int[] newShape = new int[shape.length];
        int[] newSpread = new int[shape.length];
        int[] newOffset = new int[shape.length];
        for ( int i = 0; i < shape.length; i++ ) {
            int finalI = i;
            _axisSliceBuilders[ i ] = new AxisSliceBuilder<>(
                                                shape[ i ],
                                                ( from, to, step ) -> {
                                                    if ( from < 0 && to < 0 && from > to ) {
                                                        int temp = from;
                                                        from = to;
                                                        to = temp;
                                                    }
                                                    from = ( from < 0 ) ? shape[finalI] + from : from;
                                                    to = ( to < 0 ) ? shape[finalI] + to : to;
                                                    if ( to < 0 ) to += shape[ finalI ];
                                                    if ( from < 0 ) from += shape[ finalI ];
                                                    newOffset[ finalI ] = from;
                                                    newShape[ finalI ] = ( to - from + 1 ) / step;
                                                    newSpread[ finalI ] = step;
                                                    _axisSliceBuilders[ finalI ] = null;
                                                    return this;
                                                });
        }
        _create = allowAutograd -> {
            for ( AxisSliceBuilder<V> axis : _axisSliceBuilders ) {
                if ( axis != null ) axis.resolve();
            }
            return sliceCreator.sliceOf( newShape, newOffset, newSpread, allowAutograd );
        };
    }

    /**
     *  This method returns an instance of the {@link AxisSliceBuilder} disguised by the {@link FromOrAt} interface.
     *  The {@link AxisSliceBuilder} class implements the {@link FromOrAt} interface in order to ensure
     *  that the builder methods of this API are being called in the correct order.
     *
     * @param axis The index of the axis which ought to be sliced.
     * @return An instance of the {@link AxisSliceBuilder} disguised by the {@link FromOrAt} interface.
     */
    @Override
    public FromOrAtTsr<V> axis(int axis ) {
        if ( axis >= _axisSliceBuilders.length ) throw new IllegalArgumentException("");
        return _axisSliceBuilders[ axis ];
    }

    /**
     *  This method will create and return a new slice tensor based on the
     *  provided configuration through methods like {@link AxisSliceBuilder#from(int)},
     *  {@link AxisSliceBuilder#to(int)} and {@link AxisSliceBuilder#at(int)}... <br>
     *
     * @return The slice of the tensor supplied to the constructor of this {@link SliceBuilder} instance.
     */
    @Override
    public Tsr<V> get() {
        return _create.apply(true);
    }

    @Override
    public Tsr<V> detached() {
        return _create.apply(false);
    }


}
