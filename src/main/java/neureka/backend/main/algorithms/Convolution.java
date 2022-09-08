package neureka.backend.main.algorithms;

import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.dtype.NumericType;

public final class Convolution extends AbstractFunDeviceAlgorithm<Convolution>
{
    public Convolution() {
        super("convolution");
        setIsSuitableFor(
            call ->
                call.validate()
                    .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                    .basicSuitability()
        );
    }

}
