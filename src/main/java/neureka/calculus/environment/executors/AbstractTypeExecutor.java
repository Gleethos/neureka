package neureka.calculus.environment.executors;


import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.autograd.GraphNode;
import neureka.calculus.environment.OperationType;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

public abstract class AbstractTypeExecutor<CreatorType> implements TypeExecutor
{
    protected String _operation;
    protected String _deriviation;
    protected CreatorType _creator;

    private Map<Device, Execution> _executions;

    public AbstractTypeExecutor(String operation, String deriviation, CreatorType creator)
    {
        _operation = operation;
        _deriviation = deriviation;
        _creator = creator;
        _executions = new HashMap<>();
    }


    public String getAsString(){
        return _operation;
    }
    public String getDeriviationAsString(){
        return _deriviation;
    }
    public CreatorType getCreator(){
        return _creator;
    }

    @Override
    public <T extends Execution> TypeExecutor setExecution(Device device, T execution){
        _executions.put(device, execution);
        return this;
    }

    @Override
    public <T extends Execution> T getExecution(Device device){
        return (T) _executions.get(device); // assert that result is of type T...
    }

    public Tsr reduce(TypeExecutor.ExecutionCall call, Consumer<TypeExecutor.ExecutionCall> finalExecution)
    {
        Tsr[] tsrs = call.getTensors();

        Consumer<Tsr>[] rollbacks = new Consumer[tsrs.length];
        for (int i=0; i<tsrs.length; i++) {
            if ( tsrs[i] != null && !tsrs[i].isOutsourced() ) {
                call.getDevice().add(tsrs[i]);
                rollbacks[i] = call.getDevice()::get;
            } else {
                rollbacks[i] = t->{};
            }
        }
        if ( tsrs.length > 3 )
        {
            if ( call.getDerivativeIndex() < 0 ) {
                reduce(
                        new TypeExecutor.ExecutionCall(
                                call.getDevice(),
                                new Tsr[]{tsrs[0], tsrs[1], tsrs[2]},
                                call.getDerivativeIndex(),
                                call.getType()
                        ),
                        finalExecution
                );
                Tsr[] newTsrs = Utility._offsetted(tsrs, 1);
                newTsrs[0] =  reduce(
                        new TypeExecutor.ExecutionCall(
                                call.getDevice(),
                                newTsrs,
                                call.getDerivativeIndex(),
                                call.getType()
                        ),
                        finalExecution
                );//This recursion should work!
                tsrs[0] = newTsrs[0];
            } else {
                Tsr[] newTsrs;
                switch ( call.getType().identifier() )
                {
                    case "+":
                    case "sum":
                        tsrs[0] = Tsr.Create.newTsrLike(tsrs[1]).setValue(1.0f);
                        break;

                    case "-": tsrs[0] = Tsr.Create.newTsrLike(tsrs[1]).setValue((call.getDerivativeIndex()==0)?1.0f:-1.0f);
                        break;

                    case "^":
                        newTsrs = Utility._subset(tsrs, 1,  2, tsrs.length-2);
                        if ( call.getDerivativeIndex()==0 ) {
                            newTsrs = Utility._subset(tsrs, 1,  2, tsrs.length-2);
                            newTsrs[0] =  Tsr.Create.newTsrLike(tsrs[1]);
                            Tsr exp = reduce(
                                    new TypeExecutor.ExecutionCall(
                                            call.getDevice(),
                                            newTsrs,
                                            -1,
                                            OperationType.instance("*")
                                    ),
                                    finalExecution
                            );
                            tsrs[0] = reduce(
                                    new TypeExecutor.ExecutionCall(
                                            call.getDevice(),
                                            new Tsr[]{tsrs[0], tsrs[1], exp},
                                            0,
                                            call.getType()
                                    ),
                                    finalExecution
                            );
                            exp.delete();
                        } else {
                            newTsrs[0] =  Tsr.Create.newTsrLike(tsrs[1]);
                            Tsr inner = reduce(
                                    new TypeExecutor.ExecutionCall(
                                            call.getDevice(),
                                            newTsrs,
                                            call.getDerivativeIndex()-1,
                                            OperationType.instance("*")
                                    ),
                                    finalExecution
                            );
                            Tsr exp = reduce(
                                    new TypeExecutor.ExecutionCall(
                                            call.getDevice(),
                                            new Tsr[]{Tsr.Create.newTsrLike(tsrs[1]), inner, tsrs[call.getDerivativeIndex()]},
                                            -1,
                                            OperationType.instance("*")
                                    ),
                                    finalExecution
                            );
                            tsrs[0] =  reduce(
                                    new TypeExecutor.ExecutionCall(
                                            call.getDevice(),
                                            new Tsr[]{tsrs[0], tsrs[1], exp},
                                            1,
                                            call.getType()
                                    ),
                                    finalExecution
                            );
                            inner.delete();
                            exp.delete();
                        }
                        break;
                    case "*":
                    case "prod":
                        newTsrs = Utility._without(tsrs, 1+ call.getDerivativeIndex());
                        if ( newTsrs.length > 2 ) {
                            newTsrs[0] = ( newTsrs[0] == null ) ? Tsr.Create.newTsrLike(tsrs[1]) : newTsrs[0];
                            tsrs[0] = reduce(
                                    new TypeExecutor.ExecutionCall(
                                            call.getDevice(),
                                            newTsrs,
                                            -1,
                                            OperationType.instance("*")
                                    ),
                                    finalExecution

                            );
                        } else {
                            tsrs[0] = newTsrs[1];
                        }
                        break;

                    case "/":
                        Tsr a;
                        if ( call.getDerivativeIndex() > 1 ) {
                            newTsrs = Utility._subset(tsrs, 1, 1, call.getDerivativeIndex()+1);
                            newTsrs[0] =  Tsr.Create.newTsrLike(tsrs[1]);
                            a = reduce(
                                    new TypeExecutor.ExecutionCall(
                                            call.getDevice(),
                                            newTsrs,
                                            -1,
                                            OperationType.instance("/")
                                    ),
                                    finalExecution
                            );
                        } else if ( call.getDerivativeIndex() == 1 ) a = tsrs[1];
                        else a = Tsr.Create.newTsrLike(tsrs[1], 1.0);
                        Tsr b;
                        if ( tsrs.length -  call.getDerivativeIndex() - 2  > 1 ) {
                            newTsrs = Utility._subset(tsrs, 2, call.getDerivativeIndex()+2, tsrs.length-(call.getDerivativeIndex()+2));//or (d+2)
                            newTsrs[1] =  Tsr.Create.newTsrLike(tsrs[1], 1.0);
                            newTsrs[0] = newTsrs[1];
                            b = reduce(
                                    new TypeExecutor.ExecutionCall(
                                            call.getDevice(),
                                            newTsrs,
                                            -1,
                                            OperationType.instance("/")
                                    ),
                                    finalExecution
                            );
                        } else {
                            b = Tsr.Create.newTsrLike(tsrs[1], 1.0);
                        }
                        reduce(
                                new TypeExecutor.ExecutionCall(
                                        call.getDevice(),
                                        new Tsr[]{tsrs[0], a, b},
                                        -1,
                                        OperationType.instance("*")
                                ),
                                finalExecution
                        );
                        reduce(
                                new TypeExecutor.ExecutionCall(
                                        call.getDevice(),
                                        new Tsr[]{tsrs[0], tsrs[0], tsrs[call.getDerivativeIndex()+1]},
                                        1,
                                        OperationType.instance("/")
                                ),
                                finalExecution
                        );
                        if ( call.getDerivativeIndex()== 0 ) a.delete();
                        b.delete();
                        break;
                    default: throw new IllegalStateException("Operation not found!");
                }
            }
        } else {
            switch (call.getType().identifier()) {
                case "x":
                    if (call.getDerivativeIndex() >= 0) {
                        if (call.getDerivativeIndex() == 0) tsrs[0] = tsrs[2];
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
            finalExecution.accept(call);
            //call.getDevice().execute(tsrs, call.getType(), call.getDerivativeIndex());
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


