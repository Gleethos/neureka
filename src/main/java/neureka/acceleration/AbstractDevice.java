package neureka.acceleration;

import neureka.Component;
import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.subtypes.Activation;

import java.lang.ref.Cleaner;
import java.util.function.Consumer;

public abstract class AbstractDevice implements  Device, Component<Tsr>
{
    private static final Cleaner CLEANER = Cleaner.create();

    protected abstract void _enqueue(Tsr[] tsrs, int d, OperationType type);

    protected abstract void _enqueue(Tsr t, double value, int d, OperationType type);

    @Override
    public void update(Tsr oldOwner, Tsr newOwner){
        swap(oldOwner, newOwner);
    }

    @Override
    public Device cleaning(Tsr tensor, Runnable action){
        CLEANER.register(tensor, action);
        return this;
    }

    protected void _cleaning(Object o, Runnable action){
        CLEANER.register(o, action);
    }

    @Override
    public Device execute(Tsr[] tsrs, OperationType type, int d)
    {
        if ( type.identifier().equals("<") )
        {
            int offset = ( tsrs[0] == null ) ? 1 : 0;
            _execute_recursively( new Tsr[]{tsrs[offset], tsrs[1+offset]}, OperationType.instance("idy"), -1 );
        }
        else if ( type.identifier().equals(">") )
        {
            int offset = ( tsrs[0] == null ) ? 1 : 0;
            _execute_recursively( new Tsr[]{tsrs[1+offset], tsrs[offset]}, OperationType.instance("idy"), -1 );
        }
        else
        {
            _createNewDrainTensorIn(this, tsrs, type);
            if (
                    tsrs.length == 3 && d<0 && // TODO: refactor so that 'd<0 && '
                            (
                                    tsrs[1].isVirtual() || tsrs[2].isVirtual() ||
                                    (
                                            !tsrs[1].isOutsourced() && tsrs[1].size() == 1
                                                    ||
                                            !tsrs[2].isOutsourced() && tsrs[2].size() == 1
                                    )
                            )
            ) {
                if (tsrs[2].isVirtual() || tsrs[2].size() == 1) {
                    _execute_recursively(new Tsr[]{tsrs[0], tsrs[1]}, OperationType.instance("idy"), -1);
                    _enqueue(tsrs[0], tsrs[2].value64()[0], d, type);
                } else {
                    _execute_recursively(new Tsr[]{tsrs[0], tsrs[2]}, OperationType.instance("idy"), -1);
                    _enqueue(tsrs[0], tsrs[1].value64()[0], d, type);
                }
            } else _execute_recursively(tsrs, type, d);
        }
        return this;
    }

    private Tsr _execute_recursively( Tsr[] tsrs, OperationType type, int d )
    {
        Consumer<Tsr>[] rollbacks = new Consumer[tsrs.length];
        for (int i=0; i<tsrs.length; i++) {
            if ( tsrs[i] != null && !tsrs[i].isOutsourced() ) {
                this.add(tsrs[i]);
                rollbacks[i] = this::get;
            } else {
                rollbacks[i] = t->{};
            }
        }
        if ( tsrs.length > 3 )
        {
            if ( d < 0 ) {
                _execute_recursively(new Tsr[]{tsrs[0], tsrs[1], tsrs[2]}, type, d);
                Tsr[] newTsrs = Utility._offsetted(tsrs, 1);
                newTsrs[0] =  _execute_recursively(newTsrs, type, d);//This recursion should work!
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
                            Tsr exp = _execute_recursively(newTsrs, OperationType.instance("*"), -1);
                            tsrs[0] = _execute_recursively(new Tsr[]{tsrs[0], tsrs[1], exp}, type, 0);
                            exp.delete();
                        } else {
                            newTsrs[0] =  Tsr.Create.newTsrLike(tsrs[1]);
                            Tsr inner = _execute_recursively(newTsrs, OperationType.instance("*"), d-1);
                            Tsr exp = _execute_recursively(new Tsr[]{Tsr.Create.newTsrLike(tsrs[1]), inner, tsrs[d]}, OperationType.instance("*"), -1);
                            tsrs[0] =  _execute_recursively(new Tsr[]{tsrs[0], tsrs[1], exp}, type, 1);
                            inner.delete();
                            exp.delete();
                        }
                        break;
                    case "*":
                    case "prod":
                        newTsrs = Utility._without(tsrs, 1+d);
                        if ( newTsrs.length > 2 ) {
                            newTsrs[0] = ( newTsrs[0] == null ) ? Tsr.Create.newTsrLike(tsrs[1]) : newTsrs[0];
                            tsrs[0] = _execute_recursively(newTsrs, OperationType.instance("*"), -1);
                        } else {
                            tsrs[0] = newTsrs[1];
                        }
                        break;

                    case "/":
                        Tsr a;
                        if ( d > 1 ) {
                            newTsrs = Utility._subset(tsrs, 1, 1, d+1);
                            newTsrs[0] =  Tsr.Create.newTsrLike(tsrs[1]);
                            a = _execute_recursively(newTsrs, OperationType.instance("/"), -1);
                        } else if ( d == 1 ) a = tsrs[1];
                        else a = Tsr.Create.newTsrLike(tsrs[1], 1.0);
                        Tsr b;
                        if ( tsrs.length -  d - 2  > 1 ) {
                            newTsrs = Utility._subset(tsrs, 2, d+2, tsrs.length-(d+2));//or (d+2)
                            newTsrs[1] =  Tsr.Create.newTsrLike(tsrs[1], 1.0);
                            newTsrs[0] = newTsrs[1];
                            b = _execute_recursively(newTsrs, OperationType.instance("/"), -1);
                        } else {
                            b = Tsr.Create.newTsrLike(tsrs[1], 1.0);
                        }
                        _execute_recursively(new Tsr[]{tsrs[0], a, b}, OperationType.instance("*"), -1);
                        _execute_recursively(new Tsr[]{tsrs[0], tsrs[0], tsrs[d+1]}, OperationType.instance("/"), 1);
                        if ( d == 0 ) a.delete();
                        b.delete();
                        break;
                    default: throw new IllegalStateException("Operation not found!");
                }
            }
        } else this._enqueue(tsrs, d, type);

        for ( int i = 0; i < tsrs.length; i++ ) {
            if ( tsrs[i] != null && !tsrs[i].isUndefined() ) rollbacks[i].accept(tsrs[i]);
        }
        return tsrs[0];
    }


    private static void _createNewDrainTensorIn(Device device, Tsr[] tsrs, OperationType type)
    {
        if ( tsrs[0] == null )// Creating a new tensor:
        {
            int[] shp = (type.identifier().endsWith("x"))
                    ? Tsr.Utility.Indexing.shpOfCon(tsrs[1].getNDConf().shape(), tsrs[2].getNDConf().shape())
                    : tsrs[1].getNDConf().shape();
            Tsr output = new Tsr(shp, 0.0);
            device.add(output);
            tsrs[0] = output;
        }
    }

    protected static class Utility
    {
        private Utility(){}

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
