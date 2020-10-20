package neureka.devices.storage;

import neureka.Tsr;

public interface Storage<ValueType>
{

    Storage store( Tsr<ValueType> tensor );

    Storage restore( Tsr<ValueType> tensor );

}
