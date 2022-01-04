package neureka.ndim.config.types;

import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.types.complex.ComplexDefaultNDConfiguration;

public final class ColumnMajorNDConfiguration extends ComplexDefaultNDConfiguration //:= IMMUTABLE
{
    private ColumnMajorNDConfiguration(
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        super(
            shape,
            translation,
            indicesMap,
            spread,
            offset
        );
    }

    public static NDConfiguration construct(
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        return _cached(
                new ColumnMajorNDConfiguration(
                        shape,
                        translation,
                        indicesMap,
                        spread,
                        offset
                )
        );
    }

    @Override
    public Layout getLayout() {
        return Layout.COLUMN_MAJOR;
    }

}

