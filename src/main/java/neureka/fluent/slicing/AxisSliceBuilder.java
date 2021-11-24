package neureka.fluent.slicing;


import neureka.Tsr;
import neureka.fluent.slicing.states.AxisOrGet;
import neureka.fluent.slicing.states.FromOrAt;
import neureka.fluent.slicing.states.StepsOrAxisOrGet;
import neureka.fluent.slicing.states.To;

public class AxisSliceBuilder<V> implements FromOrAt<V>, To<V>, StepsOrAxisOrGet<V>, AxisOrGet<V>
{

    interface Resolution<V> { SliceBuilder<V> resolve(int from, int to, int steps ); }

    private final Resolution<V> _then;
    private int _from;
    private int _to;
    private int _steps;

    AxisSliceBuilder( int axisSize, Resolution<V> then ) {
        _then = then;
        _from = 0;
        _to = axisSize - 1;
        _steps = 1;
    }

    /**
     *  This method returns an instance of this very {@link AxisSliceBuilder} instance
     *  disguised by the {@link To} interface.
     *  The {@link AxisSliceBuilder} class implements the {@link To} interface in order to ensure
     *  that the builder methods of this API are being called in the correct order.
     *
     * @param index The starting index of the slice for this current axis.
     * @return An instance of the {@link AxisSliceBuilder} disguised by the {@link To} interface.
     */
    @Override
    public To<V> from(int index ) {
        _from = index;
        return this;
    }

    /**
     *  This method returns an instance of this very {@link AxisSliceBuilder} instance
     *  disguised by the {@link StepsOrAxisOrGet} interface.
     *  The {@link AxisSliceBuilder} class implements the {@link StepsOrAxisOrGet} interface in order to ensure
     *  that the builder methods of this API are being called in the correct order.
     *
     * @param index The ending index of the slice for this current axis.
     * @return An instance of the {@link AxisSliceBuilder} disguised by the {@link StepsOrAxisOrGet} interface.
     */
    @Override
    public StepsOrAxisOrGet<V> to(int index ) {
        _to = index;
        return this;
    }

    /**
     *  This method returns an instance of this very {@link AxisSliceBuilder} instance
     *  disguised by the {@link AxisOrGet} interface.
     *  The {@link AxisSliceBuilder} class implements the {@link AxisOrGet} interface in order to ensure
     *  that the builder methods of this API are being called in the correct order.
     *
     * @param size The step size for the strides of the slice of the current axis.
     * @return An instance of the {@link AxisSliceBuilder} disguised by the {@link AxisOrGet} interface.
     */
    @Override
    public AxisOrGet<V> step(int size) {
        _steps = size;
        return this;
    }

    /**
     *  This method returns an instance of this very {@link AxisSliceBuilder} instance
     *  disguised by the {@link AxisOrGet} interface.
     *  The {@link AxisSliceBuilder} class implements the {@link AxisOrGet} interface in order to ensure
     *  that the builder methods of this API are being called in the correct order.
     *
     * @param index The starting and ending position for the slice of the current axis.
     * @return An instance of the {@link AxisSliceBuilder} disguised by the {@link AxisOrGet} interface.
     */
    @Override
    public AxisOrGet<V> at(int index ) {
        _from = index;
        _to = index;
        return this;
    }


    /**
     *  This method returns an instance of the {@link AxisSliceBuilder} targeted by the provided index.
     */
    @Override
    public FromOrAt<V> axis(int axis) {
       return _then.resolve(_from, _to, _steps).axis(axis);
    }

    @Override
    public Tsr<V> get() {
        return _then.resolve(_from, _to, _steps).get();
    }


    public void resolve() {
        _then.resolve(_from, _to, _steps);
    }

}