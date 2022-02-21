package neureka.ndim.config.types;

import neureka.ndim.config.types.sliced.SlicedNDConfiguration;

public final class ColumnMajorNDConfiguration extends SlicedNDConfiguration //:= IMMUTABLE
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

    public static ColumnMajorNDConfiguration construct(
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

