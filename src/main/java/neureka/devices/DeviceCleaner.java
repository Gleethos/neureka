package neureka.devices;

public interface DeviceCleaner
{
    static DeviceCleaner getInstance() {
        return new CustomDeviceCleaner();
    }

    void register( Object o, Runnable action );
}
