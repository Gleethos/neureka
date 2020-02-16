package neureka.acceleration;

import neureka.Component;
import neureka.Tsr;
import neureka.calculus.Function;
import neureka.autograd.GraphNode;

import java.lang.ref.Cleaner;
import java.lang.ref.ReferenceQueue;

public abstract class AbstractDevice implements  Device, Component<Tsr>
{
    private final static Cleaner CLEANER = Cleaner.create();
    protected ReferenceQueue _reference_queue;


    protected abstract void _enqueue(Tsr[] tsrs, int d, int f_id);

    protected abstract void _enqueue(Tsr t, double value, int d, int f_id);

    @Override
    public void update(Tsr oldOwner, Tsr newOwner){
        swap(oldOwner, newOwner);
    }

    @Override
    public Device cleaning(Tsr  tensor, Runnable action){
        CLEANER.register(tensor, action);
        return this;
    }

    @Override
    public Device execute(Tsr[] tsrs, int f_id, int d) {
        _execute(tsrs, f_id, d);
        return this;
    }

    protected void _execute(Tsr[] tsrs, int f_id, int d)
    {
        if(Function.TYPES.REGISTER(f_id).equals("<"))
        {
            int offset = (tsrs[0]==null)?1:0;
            _execute_recursively(new Tsr[]{tsrs[offset], tsrs[1+offset]}, Function.TYPES.LOOKUP("idy"), -1);
        }
        else if(Function.TYPES.REGISTER(f_id).equals(">"))
        {
            int offset = (tsrs[0]==null)?1:0;
            _execute_recursively(new Tsr[]{tsrs[1+offset], tsrs[offset]}, Function.TYPES.LOOKUP("idy"), -1);
        }
        else
        {
            _createNewDrainTensorIn(this, tsrs, f_id);
            if (
                    d<0 && tsrs.length == 3
                            &&
                            (
                                    (tsrs[1].isVirtual() || tsrs[2].isVirtual())
                                            ||
                                            (!tsrs[1].isOutsourced() && tsrs[1].size() == 1 || !tsrs[2].isOutsourced() && tsrs[2].size() == 1)
                            )
            ) {
                if (tsrs[2].isVirtual() || tsrs[2].size() == 1) {
                    _execute_recursively(new Tsr[]{tsrs[0], tsrs[1]}, Function.TYPES.LOOKUP("idy"), -1);
                    _execute_tensor_scalar(tsrs[0], tsrs[2].value64()[0], f_id, d);
                } else {
                    _execute_recursively(new Tsr[]{tsrs[0], tsrs[2]}, Function.TYPES.LOOKUP("idy"), -1);
                    _execute_tensor_scalar(tsrs[0], tsrs[1].value64()[0], f_id, d);
                }
            } else {
                _execute_recursively(tsrs, f_id, d);
            }
        }
    }

    private Tsr _execute_recursively(Tsr[] tsrs, int f_id, int d)
    {
        boolean[] notNative = new boolean[tsrs.length];
        for (int i=0; i<tsrs.length; i++) {
            if (tsrs[i]!=null && !tsrs[i].isOutsourced()) {
                this.add(tsrs[i]);
                notNative[i] = true;
            } else {
                notNative[i] = false;
            }
        }
        if(tsrs.length>3) {
            if(d<0){
                _execute_recursively(new Tsr[]{tsrs[0], tsrs[1], tsrs[2]}, f_id, d);
                Tsr[] newTsrs = _util._offsetted(tsrs, 1);
                newTsrs[0] =  _execute_recursively(newTsrs, f_id, d);//This recursion should work!
                tsrs[0] = newTsrs[0];
            } else {
                Tsr[] newTsrs;
                switch(Function.TYPES.REGISTER(f_id)){
                    case "+":
                    case "sum":
                        tsrs[0] = Tsr.Create.newTsrLike(tsrs[1]).setValue(1.0f);
                        break;
                    case "-": tsrs[0] = Tsr.Create.newTsrLike(tsrs[1]).setValue((d==0)?1.0f:-1.0f);
                        break;
                    case "^":
                        newTsrs = _util._subset(tsrs, 1,  2, tsrs.length-2);
                        if(d>0){
                            newTsrs[0] =  Tsr.Create.newTsrLike(tsrs[1]);
                            Tsr inner = _execute_recursively(newTsrs, Function.TYPES.LOOKUP("*"), d-1);
                            Tsr exp = _execute_recursively(new Tsr[]{Tsr.Create.newTsrLike(tsrs[1]), inner, tsrs[d]}, Function.TYPES.LOOKUP("*"), -1);
                            tsrs[0] =  _execute_recursively(new Tsr[]{tsrs[0], tsrs[1], exp}, f_id, 1);
                            inner.delete();
                            exp.delete();
                        } else {
                            newTsrs = _util._subset(tsrs, 1,  2, tsrs.length-2);
                            newTsrs[0] =  Tsr.Create.newTsrLike(tsrs[1]);
                            Tsr exp = _execute_recursively(newTsrs, Function.TYPES.LOOKUP("*"), -1);
                            tsrs[0] = _execute_recursively(new Tsr[]{tsrs[0], tsrs[1], exp}, f_id, 0);
                            exp.delete();
                        }
                        break;
                    case "*":
                    case "prod":
                        newTsrs = _util._without(tsrs, 1+d);
                        if(newTsrs.length>2){
                            newTsrs[0] = (newTsrs[0]==null)? Tsr.Create.newTsrLike(tsrs[1]):newTsrs[0];
                            tsrs[0] = _execute_recursively(newTsrs, Function.TYPES.LOOKUP("*"), -1);
                        } else {
                            tsrs[0] = newTsrs[1];
                        }
                        break;
                    case "/":
                        Tsr a = null;
                        if(d>1){//[0][1][2][3][4]
                            newTsrs = _util._subset(tsrs, 1, 1, d+1);
                            newTsrs[0] =  Tsr.Create.newTsrLike(tsrs[1]);
                            a = _execute_recursively(newTsrs, Function.TYPES.LOOKUP("/"), -1);
                        } else if(d==1){
                            a = tsrs[1];
                        } else if(d==0){
                            a = Tsr.Create.newTsrLike(tsrs[1], 1.0);
                        }
                        Tsr b;
                        if((tsrs.length-(d+2))>1){//  12 / 3 / 0.5 / 4 / 2
                            newTsrs = _util._subset(tsrs, 2, d+2, tsrs.length-(d+2));//or (d+2)
                            newTsrs[1] =  Tsr.Create.newTsrLike(tsrs[1], 1.0);
                            newTsrs[0] = newTsrs[1];
                            b = _execute_recursively(newTsrs, Function.TYPES.LOOKUP("/"), -1);
                        } else {
                            b = Tsr.Create.newTsrLike(tsrs[1], 1.0);
                        }
                        _execute_recursively(new Tsr[]{tsrs[0], a, b}, Function.TYPES.LOOKUP("*"), -1);
                        _execute_recursively(new Tsr[]{tsrs[0], tsrs[0], tsrs[d+1]}, Function.TYPES.LOOKUP("/"), 1);
                        if(d==0)a.delete();
                        b.delete();
                        break;
                    default:
                        throw new IllegalStateException("[AbstractDevice]: Operation not found!");
                }
            }
        } else {
            this._enqueue(tsrs, d, f_id);
        }
        for (int i=0; i<tsrs.length; i++) {
            if (notNative[i]&&tsrs[i]!=null&&!tsrs[i].isUndefined()){
                this.get(tsrs[i]);//When remove?
            }
        }
        return tsrs[0];
    }



    protected void _execute_tensor_scalar(Tsr t, double value, int f_id, int d)
    {
        if(d<0) {
            _enqueue(t, value, d, f_id);
        } else {
            /* Derivatives implementation: (values cannot be derived) */
            if(
                    Function.TYPES.REGISTER(f_id).equals("+")||
                            Function.TYPES.REGISTER(f_id).equals("-")||
                            Function.TYPES.REGISTER(f_id).equals("%")
            ){
                _execute_tensor_scalar(t, 0, Function.TYPES.LOOKUP("*"), -1);
                _execute_tensor_scalar(t, 1, Function.TYPES.LOOKUP("+"), -1);
            } else if(Function.TYPES.REGISTER(f_id).equals("^")){
                _execute_tensor_scalar(t, value-1, Function.TYPES.LOOKUP("^"), -1);
                _execute_tensor_scalar(t, value, Function.TYPES.LOOKUP("*"), -1);
            } else if(Function.TYPES.REGISTER(f_id).equals("*")||Function.TYPES.REGISTER(f_id).contains("x")){
                _execute_tensor_scalar(t, 0, Function.TYPES.LOOKUP("*"), -1);//???
                _execute_tensor_scalar(t, value, Function.TYPES.LOOKUP("+"), -1);
            } else if(Function.TYPES.REGISTER(f_id).equals("/")){
                _execute_tensor_scalar(t, 0, Function.TYPES.LOOKUP("*"), -1);
                _execute_tensor_scalar(t, 1/value, Function.TYPES.LOOKUP("+"), -1);
            }
        }
    }

    private static void _createNewDrainTensorIn(Device device, Tsr[] tsrs, int f_id){
        if(tsrs[0]==null)//Creating a new tensor:
        {
            int[] shp = (Function.TYPES.REGISTER(f_id).endsWith("x"))
                    ? Tsr.Utility.Indexing.shpOfCon(tsrs[1].shape(), tsrs[2].shape())
                    : tsrs[1].shape();
            Tsr output = new Tsr(shp, 0.0);
            device.add(output);
            tsrs[0] = output;
        }
    }
    //---

    protected static class _util
    {

        public static Tsr[] _subset(Tsr[] tsrs, int padding, int index, int offset){
            if(offset<0){
                index += offset;
                offset *= -1;
            }
            Tsr[] newTsrs = new Tsr[offset+padding];
            System.arraycopy(tsrs, index, newTsrs, padding, offset);
            return newTsrs;
        }
        public static Tsr[] _without(Tsr[] tsrs, int index){
            Tsr[] newTsrs = new Tsr[tsrs.length-1];
            for (int i=0; i<newTsrs.length; i++) newTsrs[i] = tsrs[i+((i<index)?0:1)];
            return newTsrs;
        }

        public static Tsr[] _offsetted(Tsr[] tsrs, int offset){
            Tsr[] newTsrs = new Tsr[tsrs.length-offset];
            newTsrs[0] = Tsr.Create.newTsrLike(tsrs[1]);//new Tsr(tsrs[1].shape());
            if(!tsrs[1].has(GraphNode.class)&&tsrs[1]!=tsrs[0]){//Deleting intermediate results!
                tsrs[1].delete();
                tsrs[1] = null;
            }
            if(!tsrs[2].has(GraphNode.class)&&tsrs[2]!=tsrs[0]){//Deleting intermediate results!
                tsrs[2].delete();
                tsrs[2] = null;
            }
            System.arraycopy(tsrs, 1+offset, newTsrs, 1, tsrs.length-1-offset);
            newTsrs[1] = tsrs[0];
            return newTsrs;
        }

    }

}
