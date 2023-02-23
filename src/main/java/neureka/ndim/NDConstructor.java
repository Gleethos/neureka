package neureka.ndim;

import neureka.Shape;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.types.views.virtual.VirtualNDConfiguration;

import java.util.Arrays;
import java.util.stream.Collectors;

public interface NDConstructor
{
    int getSize();

    int[] getShape();

    NDConfiguration produceNDC(boolean makeVirtual);

    static NDConstructor of(
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        return of(NDConfiguration.of(shape, translation, indicesMap, spread, offset));
    }

    static NDConstructor of( NDConfiguration ndc )
    {
        return new NDConstructor() {

            private Boolean _isVirtual = null;
            private NDConfiguration _constructedNDC = null;

            @Override public int getSize() { return ndc.size(); }
            @Override public int[] getShape() { return ndc.shape(); }
            @Override public NDConfiguration produceNDC(boolean makeVirtual) {

                if ( _isVirtual != null && _isVirtual != makeVirtual )
                    throw new IllegalStateException(
                            "The NDConfiguration of this NDConstructor has already been produced and cannot be changed anymore!");

                if ( _constructedNDC != null ) return _constructedNDC;
                _isVirtual = makeVirtual;
                _constructedNDC = makeVirtual ? of(ndc.shape()).produceNDC(true) : ndc;
                return _constructedNDC;
            }
        };
    }


    static NDConstructor of( Shape newShape ) {
        return of(newShape.toIntArray());
    }

    static NDConstructor of( int... newShape )
    {
        int size = NDConfiguration.Utility.sizeOfShape(newShape);
        if (size == 0) {
            String shape = Arrays.stream(newShape).mapToObj(String::valueOf).collect(Collectors.joining("x"));
            String message = "The provided shape '" + shape + "' must not contain zeros. Dimensions lower than 1 are not possible.";
            throw new IllegalArgumentException(message);
        }
        return new NDConstructor() {

            private Boolean _isVirtual = null;
            private NDConfiguration _constructedNDC = null;

            @Override public int getSize() { return size; }
            @Override public int[] getShape() { return newShape.clone(); }
            @Override
            public NDConfiguration produceNDC(boolean makeVirtual) {
                if ( _isVirtual != null && _isVirtual != makeVirtual )
                    throw new IllegalStateException(
                            "The NDConfiguration of this NDConstructor has already been produced and cannot be changed anymore!");

                if ( _constructedNDC != null ) return _constructedNDC;

                _isVirtual = makeVirtual;
                if (makeVirtual) _constructedNDC = VirtualNDConfiguration.construct(newShape);
                else {
                    int[] newTranslation = NDConfiguration.Layout.ROW_MAJOR.newTranslationFor(newShape);
                    int[] newSpread = new int[newShape.length];
                    Arrays.fill(newSpread, 1);
                    int[] newOffset = new int[newShape.length];
                    _constructedNDC =
                            NDConfiguration.of(
                                    newShape,
                                    newTranslation,
                                    newTranslation, // indicesMap
                                    newSpread,
                                    newOffset
                            );
                }
                return _constructedNDC;
            }
        };
    }

}
