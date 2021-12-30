package neureka.backend.standard.operations.linear.fast.concurrent;

import java.util.concurrent.ExecutorService;

public final class ProcessingService {

    public static final ProcessingService INSTANCE = new ProcessingService(DaemonPoolExecutor.INSTANCE);

    private final ExecutorService _executor;

    public ProcessingService(final ExecutorService executor) {
        super();
        _executor = executor;
    }

    public WorkScheduler.Divider divider() {
        return new WorkScheduler.Divider(_executor);
    }

}
