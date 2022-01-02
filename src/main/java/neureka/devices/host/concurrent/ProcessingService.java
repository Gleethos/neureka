package neureka.devices.host.concurrent;

import neureka.devices.host.CPU;

import java.util.concurrent.ExecutorService;

public final class ProcessingService {

    public static final ProcessingService INSTANCE = new ProcessingService(CPU.get().getExecutor().getPool());

    private final ExecutorService _executor;

    public ProcessingService(final ExecutorService executor) {
        super();
        _executor = executor;
    }

    public WorkScheduler.Divider divider() {
        return new WorkScheduler.Divider(_executor);
    }

}
