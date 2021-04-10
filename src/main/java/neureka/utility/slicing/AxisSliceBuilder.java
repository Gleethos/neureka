package neureka.utility.slicing;

import lombok.experimental.Accessors;
import neureka.Tsr;
import neureka.utility.slicing.states.FromOrAt;
import neureka.utility.slicing.states.StepsOrThen;
import neureka.utility.slicing.states.Then;
import neureka.utility.slicing.states.To;

@Accessors( fluent = true, prefix = {"_"}, chain = true )
public class AxisSliceBuilder<ValType> implements FromOrAt<ValType>, To<ValType>, StepsOrThen<ValType>
{

    interface Resolution<V> { SliceBuilder<V> resolve(int from, int to, int steps ); }

    private final Resolution<ValType> _then;
    private final int _axisSize;
    private int _from;
    private int _to;
    private int _steps = 1;

    AxisSliceBuilder( int axisSize, Resolution<ValType> then ) {
        _then = then;
        _axisSize = axisSize;
        _from = 0;
        _to = _axisSize - 1;
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
    public To<ValType> from( int index ) {
        _from = index;
        return this;
    }

    /**
     *  This method returns an instance of this very {@link AxisSliceBuilder} instance
     *  disguised by the {@link StepsOrThen} interface.
     *  The {@link AxisSliceBuilder} class implements the {@link StepsOrThen} interface in order to ensure
     *  that the builder methods of this API are being called in the correct order.
     *
     * @param index The ending index of the slice for this current axis.
     * @return An instance of the {@link AxisSliceBuilder} disguised by the {@link StepsOrThen} interface.
     */
    @Override
    public StepsOrThen<ValType> to( int index ) {
        _to = index;
        return this;
    }

    /**
     *  This method returns an instance of this very {@link AxisSliceBuilder} instance
     *  disguised by the {@link Then} interface.
     *  The {@link AxisSliceBuilder} class implements the {@link Then} interface in order to ensure
     *  that the builder methods of this API are being called in the correct order.
     *
     * @param index The step size for the strides of the slice of the current axis.
     * @return An instance of the {@link AxisSliceBuilder} disguised by the {@link Then} interface.
     */
    @Override
    public Then<ValType> steps( int index) {
        _steps = index;
        return this;
    }

    /**
     *  This method returns an instance of this very {@link AxisSliceBuilder} instance
     *  disguised by the {@link Then} interface.
     *  The {@link AxisSliceBuilder} class implements the {@link Then} interface in order to ensure
     *  that the builder methods of this API are being called in the correct order.
     *
     * @param index The starting and ending position for the slice of the current axis.
     * @return An instance of the {@link AxisSliceBuilder} disguised by the {@link Then} interface.
     */
    @Override
    public Then<ValType> at( int index ) {
        _from = index;
        _to = index;
        return this;
    }

    /**
     *  This method returns an instance of the original {@link SliceBuilder} instance.
     */
    @Override
    public SliceBuilder<ValType> then() {
        return _then.resolve(_from, _to, _steps);
    }

    @Override
    public Tsr<ValType> get() {
        return _then.resolve(_from, _to, _steps).get();
    }

}