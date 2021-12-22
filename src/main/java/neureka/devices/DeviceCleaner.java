package neureka.devices;

interface DeviceCleaner
{
    static DeviceCleaner getInstance() {
        return new CustomDeviceCleaner();
    }

    void register( Object o, Runnable action );
}
