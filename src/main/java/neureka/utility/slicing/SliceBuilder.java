package neureka.utility.slicing;

import neureka.Tsr;
import neureka.utility.slicing.states.AxisOrGet;
import neureka.utility.slicing.states.FromOrAt;

import java.util.function.Supplier;


public class SliceBuilder<ValType> implements AxisOrGet<ValType>
{
    public interface CreationCallback<V> { Tsr<V> sliceOf(int[] newShape, int[] newOffset, int[] newSpread ); }

    private final Supplier<Tsr<ValType>> _create;
    private final AxisSliceBuilder<ValType>[]  _axisSliceBuilders;

    public SliceBuilder( Tsr<ValType> toBeSliced, CreationCallback<ValType> sliceCreator )
    {
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
                                                    //while ( to < 0 ) to += shape[ finalI ];
                                                    //while ( from < 0 ) from += shape[ finalI ];
                                                    //to %= shape[ finalI ];
                                                    //from %= shape[ finalI ];
                                                    newOffset[ finalI ] = from;
                                                    newShape[ finalI ] = ( to - from + 1 ) / step;
                                                    newSpread[ finalI ] = step;
                                                    _axisSliceBuilders[ finalI ] = null;
                                                    return this;
                                                });
        }
        _create = () -> {
            for ( AxisSliceBuilder<ValType> axis : _axisSliceBuilders ) {
                if ( axis != null ) axis.resolve();
            }
            return sliceCreator.sliceOf( newShape, newOffset, newSpread );
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
    public FromOrAt<ValType> axis(int axis ) {
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
    public Tsr<ValType> get() {
        return _create.get();
    }



}
