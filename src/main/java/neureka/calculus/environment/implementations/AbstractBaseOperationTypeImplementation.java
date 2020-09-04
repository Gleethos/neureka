package neureka.calculus.environment.implementations;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.OperationTypeImplementation;

import java.util.function.Consumer;

public abstract class AbstractBaseOperationTypeImplementation<FinalType> implements OperationTypeImplementation<FinalType>
{

    @Override
    public Tsr recursiveReductionOf(
            ExecutionCall<Device> call,
            Consumer<ExecutionCall<Device>> finalExecution
    ) {
        Device device = call.getDevice();
        Tsr[] tsrs = call.getTensors();
        int d = call.getDerivativeIndex();
        OperationType type = call.getType();

        Consumer<Tsr>[] rollbacks = new Consumer[tsrs.length];
        for (int i=0; i<tsrs.length; i++) {
            if ( tsrs[i] != null && !tsrs[i].isOutsourced() ) {
                device.add(tsrs[i]);
                rollbacks[i] = device::get;
            }
            else rollbacks[i] = t -> {};
        }

        /* For the following operations with the correct arity RJAgent should do: ...
            case ("s" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
            case ("d" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
            case ("p" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
            case ("m" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
            case ">": tsrs = new Tsr[]{tsrs[1], tsrs[0]};
         */
        /*
            Below is the core lambda of recursive preprocessing
            which is defined for each OperationImplementation individually :
         */
        Tsr result = handleRecursivelyAccordingToArity(call, c -> recursiveReductionOf( c, finalExecution ));
        if ( result == null ) {
            finalExecution.accept(
                    new ExecutionCall<>( device, call.getTensors(), d, type )
            );
        }
        else return result;


        for ( int i = 0; i < tsrs.length; i++ ) {
            if ( tsrs[i] != null && !tsrs[i].isUndefined() ) rollbacks[i].accept(tsrs[i]);
        }
        return tsrs[0];
    }


}
