package neureka.devices;

public interface DeviceCleaner
{
    DeviceCleaner INSTANCE = new CustomDeviceCleaner();

    static DeviceCleaner getNewInstance() { return new CustomDeviceCleaner(); }

    void register( Object o, Runnable action );
}
