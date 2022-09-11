package neureka.devices.host;

import neureka.backend.api.BackendExtension;
import neureka.backend.api.ini.BackendLoader;
import neureka.backend.api.ini.BackendRegistry;

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


    }

}
