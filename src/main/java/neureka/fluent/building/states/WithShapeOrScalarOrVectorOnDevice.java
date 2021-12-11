package neureka.fluent.building.states;

import neureka.devices.Device;

public interface WithShapeOrScalarOrVectorOnDevice<V> extends WithShapeOrScalarOrVector<V> {

    WithShapeOrScalarOrVector<V> on( Device<V> device );

}
