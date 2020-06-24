package neureka.calculus.environment.executors;

import neureka.Tsr;
import neureka.calculus.environment.Type;

public class Operation  extends AbstractTypeExecutor<Operation, Type.OperatorCreator>
{
    public Operation(String strOperation, String strDeriviation, Type.OperatorCreator creator){
        super(strOperation, strDeriviation, creator);
    }

    @Override
    public boolean canHandle(ExecutionCall call) {
        int size = (call.getTensors()[0]==null)?call.getTensors()[1].size():call.getTensors()[0].size();
        for ( Tsr t : call.getTensors() ) if ( t!=null && t.size() != size ) return false;
        return true;
    }
}