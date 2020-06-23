package neureka.calculus.environment.executors;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.calculus.environment.Type;

public interface TypeExecutor
{
    public class ExecutionCall
    {
        private Device _device;
        private Tsr[] _tsrs;
        private int _d;
        public ExecutionCall(Device device, Tsr[] tsrs, int d){
            _device = device;
            _tsrs = tsrs;
            _d = d;
        }
        public Device getDevice(){return _device;}
        public Tsr[] getTensors(){return _tsrs;}
        public int getDerivativeIndex(){return _d;}
    }
    interface OperationPreprocessor
    {
        ExecutionCall process(ExecutionCall call);
    }
    //OperationPreprocessor getPreprocessor();
    //void setPreprocessor(OperationPreprocessor processor);




    boolean canHandle(ExecutionCall call);

    void handle(ExecutionCall call);


}
