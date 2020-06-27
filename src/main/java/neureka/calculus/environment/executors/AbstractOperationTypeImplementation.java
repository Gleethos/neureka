package neureka.calculus.environment.executors;


import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.autograd.GraphNode;
import neureka.calculus.environment.ExecutorFor;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.OperationTypeImplementation;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

public abstract class AbstractOperationTypeImplementation<FinalType, CreatorType> implements OperationTypeImplementation<FinalType>
{
    protected CreatorType _creator;

    protected Map<Class<ExecutorFor<Device>>, ExecutorFor<Device>> _executions;

    public AbstractOperationTypeImplementation(CreatorType creator)
    {
        _creator = creator;
        _executions = new HashMap<>();
    }

    public CreatorType getCreator(){
        return _creator;
    }

    @Override
    public <D extends Device, E extends ExecutorFor<D>> FinalType setExecution(Class<E> deviceClass, E execution){
        _executions.put((Class<ExecutorFor<Device>>) deviceClass, (ExecutorFor<Device>) execution);
        return (FinalType) this;
    }

    @Override
    public <D extends Device, E extends ExecutorFor<D>> E getExecution(Class<E> deviceClass){
        return (E) _executions.get(deviceClass); // assert that result is of type T...
    }

    //@Override
    //ExecutionCall<Device> fitArguments(ExecutionCall<Device> call)
    //{
    //    return null;
    //}
    
    private Tsr reduce(
            Device device,
            Tsr[] tsrs,
            OperationType type,
            int d,
            Consumer<OperationTypeImplementation.ExecutionCall<Device>> finalExecution
    ) {
        return reduce (
                new ExecutionCall<Device>( device, tsrs, d, type ), finalExecution
        );
    }

    @Override
    public Tsr reduce (
            OperationTypeImplementation.ExecutionCall<Device> call,
            Consumer<OperationTypeImplementation.ExecutionCall<Device>> finalExecution
    ) {
        Device device = call.getDevice();
        ExecutorFor<Device> executorFor = call.getExecutor().getExecution((Class<Device>) device.getClass());

        //assert execution!=null;

        Tsr[] tsrs = call.getTensors();
        int d = call.getDerivativeIndex();
        OperationType type = call.getType();

        Consumer<Tsr>[] rollbacks = new Consumer[tsrs.length];
        for (int i=0; i<tsrs.length; i++) {
            if ( tsrs[i] != null && !tsrs[i].isOutsourced() ) {
                device.add(tsrs[i]);
                rollbacks[i] = device::get;
            } else {
                rollbacks[i] = t -> {};
            }
        }
        if ( tsrs.length > 3 )
        {
            if ( d < 0 ) {
                reduce(device, new Tsr[]{tsrs[0], tsrs[1], tsrs[2]}, type, d, finalExecution);
                Tsr[] newTsrs = Utility._offsetted(tsrs, 1);
                newTsrs[0] =  reduce(device, newTsrs, type, d, finalExecution);//This recursion should work!
                tsrs[0] = newTsrs[0];
            } else {
                Tsr[] newTsrs;
                switch ( type.identifier() )
                {
                    case "+":
                    case "sum":
                        tsrs[0] = Tsr.Create.newTsrLike(tsrs[1]).setValue(1.0f);
                        break;

                    case "-": tsrs[0] = Tsr.Create.newTsrLike(tsrs[1]).setValue((d==0)?1.0f:-1.0f);
                        break;

                    case "^":
                        newTsrs = Utility._subset(tsrs, 1,  2, tsrs.length-2);
                        if ( d==0 ) {
                            newTsrs = Utility._subset(tsrs, 1,  2, tsrs.length-2);
                            newTsrs[0] =  Tsr.Create.newTsrLike(tsrs[1]);
                            Tsr exp = reduce(device, newTsrs, OperationType.instance("*"), -1, finalExecution);
                            tsrs[0] = reduce(device, new Tsr[]{tsrs[0], tsrs[1], exp}, type, 0, finalExecution);
                            exp.delete();
                        } else {
                            newTsrs[0] =  Tsr.Create.newTsrLike(tsrs[1]);
                            Tsr inner = reduce(device, newTsrs, OperationType.instance("*"), d-1, finalExecution);
                            Tsr exp = reduce(device, new Tsr[]{Tsr.Create.newTsrLike(tsrs[1]), inner, tsrs[d]}, OperationType.instance("*"), -1, finalExecution);
                            tsrs[0] =  reduce(device, new Tsr[]{tsrs[0], tsrs[1], exp}, type, 1, finalExecution);
                            inner.delete();
                            exp.delete();
                        }
                        break;
                    case "*":
                    case "prod":
                        newTsrs = Utility._without(tsrs, 1+d);
                        if ( newTsrs.length > 2 ) {
                            newTsrs[0] = ( newTsrs[0] == null ) ? Tsr.Create.newTsrLike(tsrs[1]) : newTsrs[0];
                            tsrs[0] = reduce(device, newTsrs, OperationType.instance("*"), -1, finalExecution);
                        } else {
                            tsrs[0] = newTsrs[1];
                        }
                        break;

                    case "/":
                        Tsr a;
                        if ( d > 1 ) {
                            newTsrs = Utility._subset(tsrs, 1, 1, d+1);
                            newTsrs[0] =  Tsr.Create.newTsrLike(tsrs[1]);
                            a = reduce(device, newTsrs, OperationType.instance("/"), -1, finalExecution);
                        } else if ( d == 1 ) a = tsrs[1];
                        else a = Tsr.Create.newTsrLike(tsrs[1], 1.0);
                        Tsr b;
                        if ( tsrs.length -  d - 2  > 1 ) {
                            newTsrs = Utility._subset(tsrs, 2, d+2, tsrs.length-(d+2));//or (d+2)
                            newTsrs[1] =  Tsr.Create.newTsrLike(tsrs[1], 1.0);
                            newTsrs[0] = newTsrs[1];
                            b = reduce(device, newTsrs, OperationType.instance("/"), -1, finalExecution);
                        } else {
                            b = Tsr.Create.newTsrLike(tsrs[1], 1.0);
                        }
                        reduce(device, new Tsr[]{tsrs[0], a, b}, OperationType.instance("*"), -1, finalExecution);
                        reduce(device, new Tsr[]{tsrs[0], tsrs[0], tsrs[d+1]}, OperationType.instance("/"), 1, finalExecution);
                        if ( d == 0 ) a.delete();
                        b.delete();
                        break;
                    default: throw new IllegalStateException("Operation not found!");
                }
            }
        } else {
            switch (type.identifier()) {
                case "x":
                    if (d >= 0) {
                        if (d == 0) tsrs[0] = tsrs[2];
                        else tsrs[0] = tsrs[1];
                        return tsrs[0];
                    } else tsrs = new Tsr[]{tsrs[0], tsrs[1], tsrs[2]};
                    break;
                case ("x" + ((char) 187)):
                    tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
                    break;
                case ("a" + ((char) 187)):
                    tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
                    break;
                case ("s" + ((char) 187)):
                    tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
                    break;
                case ("d" + ((char) 187)):
                    tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
                    break;
                case ("p" + ((char) 187)):
                    tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
                    break;
                case ("m" + ((char) 187)):
                    tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
                    break;
                case ">":
                    tsrs = new Tsr[]{tsrs[1], tsrs[0]};
                    break;
            }
            //this._enqueue(tsrs, d, type);
            finalExecution.accept(
                    new ExecutionCall( device, tsrs, d, type)
            );
        }

        for ( int i = 0; i < tsrs.length; i++ ) {
            if ( tsrs[i] != null && !tsrs[i].isUndefined() ) rollbacks[i].accept(tsrs[i]);
        }
        return tsrs[0];
    }

    protected static class Utility
    {
        public static Tsr[] _subset(Tsr[] tsrs, int padding, int index, int offset) {
            if ( offset < 0 ) {
                index += offset;
                offset *= -1;
            }
            Tsr[] newTsrs = new Tsr[offset+padding];
            System.arraycopy(tsrs, index, newTsrs, padding, offset);
            return newTsrs;
        }
        public static Tsr[] _without(Tsr[] tsrs, int index){
            Tsr[] newTsrs = new Tsr[tsrs.length-1];
            for ( int i = 0; i < newTsrs.length; i++ ) newTsrs[i] = tsrs[i+( ( i < index )? 0 : 1 )];
            return newTsrs;
        }

        public static Tsr[] _offsetted(Tsr[] tsrs, int offset){
            Tsr[] newTsrs = new Tsr[tsrs.length-offset];
            newTsrs[0] = Tsr.Create.newTsrLike(tsrs[1]);
            if ( !tsrs[1].has(GraphNode.class ) && tsrs[1] != tsrs[0] ) {//Deleting intermediate results!
                tsrs[1].delete();
                tsrs[1] = null;
            }
            if ( !tsrs[2].has(GraphNode.class) && tsrs[2] != tsrs[0] ) {//Deleting intermediate results!
                tsrs[2].delete();
                tsrs[2] = null;
            }
            System.arraycopy(tsrs, 1+offset, newTsrs, 1, tsrs.length-1-offset);
            newTsrs[1] = tsrs[0];
            return newTsrs;
        }

    }






}


