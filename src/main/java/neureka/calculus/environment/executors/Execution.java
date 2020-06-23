package neureka.calculus.environment.executors;

public interface Execution
{

    boolean canExecute(TypeExecutor.ExecutionCall call);

    void execute(TypeExecutor.ExecutionCall call);

}
