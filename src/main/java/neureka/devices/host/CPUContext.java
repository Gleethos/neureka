package neureka.devices.host;

import neureka.backend.api.BackendExtension;
import neureka.backend.api.ini.BackendLoader;
import neureka.backend.api.ini.ReceiveForDevice;
import neureka.backend.main.algorithms.BiElementWise;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.implementations.broadcast.*;
import neureka.backend.main.implementations.elementwise.CPUBiElementWiseAddition;
import neureka.backend.main.implementations.elementwise.CPUBiElementWisePower;
import neureka.backend.main.implementations.elementwise.CPUBiElementWiseSubtraction;
import neureka.backend.main.operations.operator.Addition;
import neureka.backend.main.operations.operator.Power;
import neureka.backend.main.operations.operator.Subtraction;

public class CPUContext implements BackendExtension
{
    @Override
    public DeviceOption find(String searchKey) {
        if ( searchKey.equalsIgnoreCase("cpu")  ) new DeviceOption( CPU.get(), 1f );
        if ( searchKey.equalsIgnoreCase("jvm")  ) new DeviceOption( CPU.get(), 1f );
        if ( searchKey.equalsIgnoreCase("java") ) new DeviceOption( CPU.get(), 1f );
        return new DeviceOption( CPU.get(), 0f );
    }

    @Override
    public void dispose() { CPU.get().dispose(); }

    @Override
    public BackendLoader getLoader() {
        return registry -> _load( registry.forDevice(CPU.class) );
    }

    private void _load( ReceiveForDevice<CPU> receive )
    {
        receive.forOperation( Power.class )
                .set( Scalarization.class, context -> new CPUScalaBroadcastPower() )
                .set( Broadcast.class,     context -> new CPUBroadcastPower() )
                .set( BiElementWise.class, context -> new CPUBiElementWisePower() );

        receive.forOperation( Addition.class )
                .set( Scalarization.class, context -> new CPUScalarBroadcastAddition() )
                .set( Broadcast.class,     context -> new CPUBroadcastAddition() )
                .set( BiElementWise.class, context -> new CPUBiElementWiseAddition() );

        receive.forOperation( Subtraction.class )
                .set( Scalarization.class, context -> new CPUScalarBroadcastSubtraction() )
                .set( Broadcast.class,     context -> new CPUBroadcastSubtraction() )
                .set( BiElementWise.class, context -> new CPUBiElementWiseSubtraction() );
    }

}
