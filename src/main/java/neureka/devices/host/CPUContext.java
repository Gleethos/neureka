package neureka.devices.host;

import neureka.backend.api.BackendExtension;
import neureka.backend.api.ini.BackendLoader;
import neureka.backend.api.ini.BackendRegistry;
import neureka.backend.main.algorithms.BiElementWise;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.implementations.broadcast.CPUBroadcastPower;
import neureka.backend.main.implementations.broadcast.CPUScalaBroadcastPower;
import neureka.backend.main.implementations.broadcast.CPUScalarBroadcastAddition;
import neureka.backend.main.implementations.elementwise.CPUBiElementWiseAddition;
import neureka.backend.main.implementations.elementwise.CPUBiElementWisePower;
import neureka.backend.main.operations.operator.Addition;
import neureka.backend.main.operations.operator.Power;

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
        return this::_load;
    }

    private void _load( BackendRegistry registry )
    {

        registry.forDevice( CPU.class )
                .andOperation( Power.class )
                .set( Scalarization.class, context -> new CPUScalaBroadcastPower() )
                .set( Broadcast.class, context -> new CPUBroadcastPower() )
                .set( BiElementWise.class, context -> new CPUBiElementWisePower() );
//
        //registry.forDevice( CPU.class )
        //        .andOperation( Addition.class )
        //        .set( Scalarization.class, context -> new CPUScalarBroadcastAddition() )
        //        .set( Broadcast.class,     context -> new CPUBiElementWiseAddition() );

    }

}
