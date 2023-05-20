package neureka.fluent.building.states;

import neureka.devices.Device;

public interface WithShapeOrScalarOrVectorOnDevice<V> extends WithShapeOrScalarOrVectorTensor<V> {

    /**
     *  Use this to specify the type onto which the tensor should be stored.
     *
     * @param device The {@link Device} which should host the tensor built by this builder.
     * @return The next fluent builder API step, which requires the definition of the shape of the tensor.
     */
    WithShapeOrScalarOrVectorTensor<V> on( Device<V> device );

}
