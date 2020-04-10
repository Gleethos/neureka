package neureka;

import neureka.abstraction.AbstractNDArray;
import neureka.acceleration.host.HostCPU;
import neureka.acceleration.Device;
import neureka.framing.IndexAlias;
import neureka.framing.Relation;
import neureka.calculus.Function;
import neureka.calculus.factory.assembly.FunctionBuilder;
import neureka.autograd.GraphNode;
import neureka.autograd.JITProp;
import neureka.optimization.Optimizer;
import neureka.utility.DataHelper;

import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

public class Tsr extends AbstractNDArray<Tsr>
{
    static{ _CPU = HostCPU.instance(); }
    
    /**
     *  Default device (host cpu)
     */
    private static final Device _CPU;
    
    /**
     *  Flag Fields
     */
    private int _flags = 0;

    private static final int RQS_GRADIENT_MASK = 1;
    private static final int IS_OUTSOURCED_MASK = 2;
    private static final int IS_VIRTUAL_MASK = 4;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public Tsr setRqsGradient(boolean rqsGradient) {
        if (rqsGradient() != rqsGradient && !rqsGradient) this.remove(Tsr.class);
        _setRqsGradient(rqsGradient);
        return this;
    }

    public boolean rqsGradient() {
        return (_flags & RQS_GRADIENT_MASK) == RQS_GRADIENT_MASK;
    }

    protected void _setRqsGradient(boolean rqsGradient) {
        if (rqsGradient() != rqsGradient) {
            if (rqsGradient) _flags += RQS_GRADIENT_MASK;
            else _flags -= RQS_GRADIENT_MASK;
        }
    }

    public Tsr setIsOutsourced(boolean isOutsourced) {
        _setIsOutsourced(isOutsourced);
        if (isOutsourced) {
            _value = null;
        } else if (
                !forComponent(
                    Device.class,
                    d ->{
                        if (((Device)d).has(this)) ((Device)d).get(this);
                        this.remove(Device.class);
                        forComponent(
                            Tsr.class,
                            gradient ->
                            ((Tsr) gradient).forComponent(Device.class, gd ->{
                                if (((Device)gd).has((Tsr)gradient)) ((Device)gd).get((Tsr)gradient);
                                ((Tsr) gradient).remove(Device.class);
                            })
                        );
                    }
                ) && _value==null
        ){
            setIsVirtual(true);
        }
        return this;
    }

    public boolean isOutsourced() {
        return (_flags & IS_OUTSOURCED_MASK) == IS_OUTSOURCED_MASK;
    }

    protected void _setIsOutsourced(boolean isOutsourced) {
        if (isOutsourced() != isOutsourced) {
            if (isOutsourced) _flags += IS_OUTSOURCED_MASK;
            else _flags -= IS_OUTSOURCED_MASK;
        }
    }

    public Tsr setIsVirtual(boolean isVirtual) {
        if (isVirtual() != isVirtual) {
            if (this.isOutsourced()) {
                if (!isVirtual) _setIsVirtual(false);
            } else {
                double v = (_value==null) ? 0 : ((this.is64())?((double[])_value)[0]:((float[])_value)[0]);
                if (isVirtual) {
                    _value = new double[]{v};
                    Relation parent = (Relation)find(Relation.class);
                    if (parent!=null) parent.foreachChild((c)->c._value=_value);
                } else {
                    _value = (this.is64())?new double[this.size()]:new float[this.size()];
                    int length = (this.is64())?((double[])_value).length:((float[])_value).length;
                    for (int i = 0; i < length; i++) {
                        if (this.is64()) ((double[])_value)[i] = v;
                        else ((float[])_value)[i] = (float)v;
                    }
                }
                _setIsVirtual(isVirtual);
            }
        } else if (isVirtual && _value==null) _value = new double[]{0};
        return this;
    }

    public boolean isVirtual() {
        return (_flags & IS_VIRTUAL_MASK) == IS_VIRTUAL_MASK;
    }

    protected void _setIsVirtual(boolean isVirtual) {
        if (isVirtual() != isVirtual) {
            if (isVirtual) _flags += IS_VIRTUAL_MASK;
            else _flags -= IS_VIRTUAL_MASK;
        }
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /**
     * This method is executed when a new Component is added to the tensor.
     * The public add method is implemented in the super class
     * 'AbstractComponentOwner' from which this class inherits.
     * In this super class the component logic is implemented.
     *
     * @param newComponent A component used to access features. (GraphNode, IndexAlias, Relation, int[], ...)
     * @return The unchanged object or maybe in future versions: null (component rejected)
     */
    @Override
    protected Object _addOrReject(Object newComponent){
        if (newComponent instanceof Device && !((Device)newComponent).has(this)){
            ((Device)newComponent).add(this);
        }
        return newComponent;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // HIGH LEVEL PROPERTIES :

    public boolean isEmpty() {
        return _value == null && !this.isOutsourced();
    }

    public boolean isUndefined() {
        return _shape == null;
    }

    public boolean isSlice(){
        Relation child = (Relation)find(Relation.class);
        return (child != null && child.hasParent());
    }

    public int sliceCount(){
        Relation child = (Relation)find(Relation.class);
        return (child!=null) ? child.childCount() : 0;
    }

    public boolean isSliceParent(){
        Relation parent = (Relation)find(Relation.class);
        return (parent!=null && parent.hasChildren());
    }

    public boolean belongsToGraph() {
        return this.has(GraphNode.class);
    }

    public boolean isLeave() {
        return (!this.has(GraphNode.class)) || ((GraphNode) this.find(GraphNode.class)).isLeave();
    }

    public boolean isBranch() {
        return !this.isLeave();
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Direct Access to component (Device)

    /**
     * @return The device on which this tensor is stored or 'CPU' if it is not outsourced.
     */
    public Device device() {
        if (this.isOutsourced()) return (Device) this.find(Device.class);
        return _CPU;
    }

    /**
     *
     * @return The graph node of the computation graph to which this tensor belongs or null if not part of a graph.
     */
    public GraphNode graphNode(){
        return (GraphNode) find(GraphNode.class);
    }

    /**
     *
     * @return Custom IndexAlias object.
     */
    public IndexAlias index(){
        return (IndexAlias) find(IndexAlias.class);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    protected Tsr _become(Tsr tensor) {
        if (tensor==null) return this;
        _value = tensor._value;
        _shape = tensor._shape;
        _idxmap = tensor._idxmap;
        _translation = tensor._translation;
        _spread = tensor._spread;
        _offset = tensor._offset;
        _components = Collections.synchronizedList(new ArrayList<Object>());//tensor._components
        _flags = tensor._flags;
        if (tensor._components!=null) {//Inform components about their new owner:
            _components.addAll(tensor._components);
            List<Object> snapshot = new ArrayList<>(tensor._components);
            for (Object o : snapshot) {
                if (o instanceof Component) ((Component<Tsr>) o).update(tensor, this);
            }
        }
        tensor._value = null;
        tensor._shape = null;
        tensor._idxmap = null;
        tensor._translation = null;
        tensor._spread = null;
        tensor._offset = null;
        tensor._components = null;
        tensor._flags = -1;
        return this;
    }

    public Tsr delete() {
        forComponent(Device.class, d ->((Device)d).rmv(this));
        forComponent(GraphNode.class, n ->{
            if (((GraphNode)n).isUsedAsDerivative()) {
                throw new IllegalStateException("Trying to delete a tensor which is part of a function graph and used as derivative!");
            }
        });
        _flags = -1;
        _value = null;
        _shape = null;
        _translation = null;
        _idxmap = null;
        forComponent(Tsr.class, g ->((Tsr)g).delete());
        _components = null;
        return this;
    }
    
    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /**
     * @param newShape
     * @return
     */
    protected void _configureFromNewShape(int[] newShape) {
        int size = Utility.Indexing.szeOfShp(newShape);
        _value = (_value==null) ? new double[size] : _value;
        int length = (this.is64())?((double[])_value).length:((float[])_value).length;
        if (size != length && !this.isVirtual()) {
            throw new IllegalArgumentException("Size of shape does not match stored value64!");
        }
        _shape = _cached(newShape);
        _translation = _cached(Utility.Indexing.newTlnOf(newShape));
        _idxmap = _translation;
        _offset = _cached(new int[newShape.length]);
        _spread = new int[newShape.length];
        Arrays.fill(_spread, 1);
        _cached(_spread);
    }


    //CONSTRUCTION :
    //=========================

    public Tsr(){}

    //Generic construction: (Groovy, Scala, ...)
    public Tsr(Object arg){
        _construct(new Object[]{arg});
    }

    public Tsr(List arg1, String arg2){
        if ((arg1).get(0) instanceof Integer) {
            List<Integer> shape = arg1;
            int[] shp = new int[shape.size()];
            for (int i=0; i<shp.length; i++) shp[i] = shape.get(i);
            _construct(shp, arg2);
        } else if ((arg1).get(0) instanceof Tsr) {
            _construct(((List<Tsr>)arg1).toArray(new Tsr[0]), arg2, true);
        }
    }

    public Tsr(List<Integer> shape, List range){
        int[] shp = new int[shape.size()];
        for(int i=0; i<shp.length; i++) shp[i] = shape.get(i);
        double[] value = new double[Utility.Indexing.szeOfShp(shp)];
        for(int i=0; i<value.length; i++){
            if(range.get(i%range.size()) instanceof BigDecimal){
                value[i] = ((BigDecimal)range.get(i%range.size())).doubleValue();
            } else {
                value[i] = (Integer)range.get(i%range.size());
            }
        }
        _construct(shp, value);
    }

    public Tsr(List conf){
        boolean isNatural = !(conf.size() > 64);
        for(Object e : conf){
            if(!isNatural) break;
            double asNum = (e instanceof BigDecimal)? ((BigDecimal)e).doubleValue() : (e instanceof Double) ?(Double)e : (Integer)e;
            isNatural = asNum % 1 == 0;
        }
        if(isNatural){
            int[] shape = new int[conf.size()];
            for(int i=0; i<shape.length; i++) {
                shape[i] = (conf.get(i) instanceof BigDecimal)
                        ? ((BigDecimal)conf.get(i)).intValue() :
                            (conf.get(i) instanceof Double)
                                    ?((Double)conf.get(i)).intValue()
                                    :((Integer)conf.get(i));
            }
            _construct(shape);
        } else {
            double[] value = new double[conf.size()];
            for(int i=0; i<value.length; i++) {
                value[i] = (conf.get(i) instanceof BigDecimal)
                        ? ((BigDecimal)conf.get(i)).doubleValue() :
                            (conf.get(i) instanceof Double)
                                ?((Double)conf.get(i)).doubleValue()
                                :((Integer)conf.get(i));
            }
            _construct(new int[]{conf.size()}, value);
        }

    }

    public Tsr(Object arg1, Object arg2) {
        _construct(new Object[]{arg1, arg2});
    }
    public Tsr(Object arg1, Object arg2, Object arg3){
        _construct(new Object[]{arg1, arg2, arg3});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4){
        _construct(new Object[]{arg1, arg2, arg3, arg4});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4, Object arg5){
        _construct(new Object[]{arg1, arg2, arg3, arg4, arg5});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6){
        _construct(new Object[]{arg1, arg2, arg3, arg4, arg5, arg6});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7){
        _construct(new Object[]{arg1, arg2, arg3, arg4, arg5, arg6, arg7});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7, Object arg8){
        _construct(new Object[]{arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7, Object arg8, Object arg9){
        _construct(new Object[]{arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7, Object arg8, Object arg9, Object arg10){
        _construct(new Object[]{arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10});
    }

    public Tsr(int[] shape, String seed){
        _construct(shape, seed);
    }

    private void _construct(int[] shape, String seed) {
        _construct(shape);
        _value = DataHelper.seededDoubleArray((double[])_value, seed);
    }

    private int[] _intArray(Object[] arg) {
        int length = arg.length;
        int[] array = new int[length];
        for (int i=0; i<length; i++){
            if(arg[i] instanceof Double) array[i] = ((Double) arg[i]).intValue();
            else array[i] = (Integer) arg[i];
        }
        return array;
    }

    private double[] _doubleArray(Object[] arg) {
        int length = arg.length;
        double[] array = new double[length];
        for (int i=0; i<length; i++){
            if (arg[i] instanceof Integer) array[i] = (Integer) arg[i];
            else if (arg[i] instanceof Double) array[i] = (Double) arg[i];
            else if (arg[i] instanceof BigDecimal) array[i] = ((BigDecimal)arg[i]).doubleValue();
        }
        return array;
    }

    public Tsr(Object[] args){
        _construct(args);
    }

    private void _construct(Object[] args) {
        if (args==null || args.length==0) return;
        if (args.length==1) return;
        args[0] = (args[0] instanceof ArrayList) ? ((ArrayList)args[0]).toArray() : args[0];
        args[1] = (args[1] instanceof ArrayList) ? ((ArrayList)args[1]).toArray() : args[1];
        if (args[0] instanceof Object[]){
            if (((Object[])args[0])[0] instanceof Integer || ((Object[])args[0])[0] instanceof Double) {
                args[0] = _intArray((Object[])args[0]);
            }
        }
        if (args[1] instanceof Object[]) {
            if (((Object[])args[1])[0] instanceof Integer) args[1] = _doubleArray((Object[]) args[1]);
            else if (((Object[])args[1])[0] instanceof BigDecimal) args[1] = _doubleArray((Object[]) args[1]);
        }
        //CASES:
        if (args[0] instanceof int[] && (args[1] instanceof Double || args[1] instanceof Integer)) {
            args[1] = (args[1] instanceof Integer)?((Integer)args[1]).doubleValue():args[1];
            _construct((int[])args[0], (Double) args[1]);
            return;
        } else if (args[0] instanceof int[] && args[1] instanceof double[]) {
            _construct((int[])args[0], (double[])args[1]);
            return;
        }
        //EQUATION:
        boolean containsString = false;
        int numberOfTensors = 0;
        ArrayList<Tsr> list = new ArrayList<>();
        for (Object o : args) {
            containsString = (o instanceof String) || containsString;
            if (o instanceof Tsr && !list.contains(o)) {
                list.add( (Tsr)o );
                numberOfTensors++;
            }
        }
        boolean doAD = true;
        Tsr[] tsrs = new Tsr[numberOfTensors];
        StringBuilder f = new StringBuilder();
        int ti=0;
        for (Object o : args) {
            if (list.contains(o)){
                tsrs[ti] = ((Tsr)o);
                f.append("I[").append(ti).append("]");
                ti++;
            } else if (o instanceof  String) f.append((String) o);
            else if (o instanceof  Boolean) doAD = (Boolean)o;
        }
        _construct(tsrs, f.toString(), doAD);
    }

    public Tsr(double value){
        _construct(new int[]{1}, value);
    }

    public Tsr(int[] shape) {
        _construct(shape);
    }

    private void _construct(int[] shape){
        _value = new double[Utility.Indexing.szeOfShp(shape)];
        this._configureFromNewShape(shape);
    }

    public Tsr(int[] shape, double value) {
        _construct(shape, value);
    }

    private void _construct(int[] shape, double value){
        int size = Utility.Indexing.szeOfShp(shape);
        _value = new double[1];
        this.setIsVirtual( size > 1 );
        this._configureFromNewShape(shape);
        ((double[])_value)[0] = value;
    }

    public Tsr(int[] shape, double[] value) {
         _construct(shape, value);
    }

    private void _construct(int[] shape, double[] value) {
        int size = Utility.Indexing.szeOfShp(shape);
        if (size!=value.length) {
            double[] newValue = new double[size];
            for(int i=0; i<newValue.length; i++) newValue[i] = value[i%value.length];
            _value = newValue;
        } else _value = value;
        this._configureFromNewShape(shape);
    }

    /**
     * @param tensor which acts as template for this new tensor.
     */
    public Tsr(Tsr tensor, boolean cpy) {//TODO: Remove this and make it be replaced by getAt([...])
        _value = tensor._value;//(tensor.is64()) ? new double[tensor.size()] : new float[tensor.size()];
        _components = null;
        //int length = (tensor.is64()) ? ((double[])_value).length : ((float[])_value).length;
        //if(cpy) {
        //    if (tensor.is64()) {
        //        double[] value = tensor.value64();
        //        System.arraycopy(value, 0, _value, 0, length);
        //    } else {
        //        float[] value = tensor.value32();
        //        System.arraycopy(value, 0, _value, 0, length);
        //    }
        //}
        this._configureFromNewShape(tensor.shape());
    }


    //TRACKED COMPUTATION :
    //=========================
    public Tsr(Tsr tensor, String operation) {
        if (tensor == null) return;
        _construct(new Tsr[]{tensor}, operation, true);
    }

    public Tsr(Tsr[] tensors, String operation) {
        _construct(tensors, operation, true);
    }

    public Tsr(Tsr[] tensors, String operation, boolean doAD) {
        _construct(tensors, operation, doAD);
    }

    private void _construct(Tsr[] tensors, String operation, boolean doAD) {
        if (tensors == null || tensors.length == 0 || tensors[0] == null) return;
        Tsr result = Function.Setup.commit(this, tensors, operation, doAD);
        this._become(result);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //MODIFICATION :
    //=========================

    /**
     *
     * @param error A tensor which is back-propagated to gradients. Must match the size og this tensor.
     * @return The tensor on which this method was called.
     */
    public Tsr backward(Tsr error) {
        if (!forComponent(GraphNode.class, node->((GraphNode)node).backward(error)) && this.rqsGradient()) {
            addToGradient(error);
        }
        return this;
    }

    /**
     *
     * @param value
     * @return
     */
    public Tsr backward(double value) {
        backward(new Tsr(_shape, value));
        return this;
    }

    public void applyGradient() {
        forComponent(JITProp.class, jit->((JITProp)jit).execute());
        forComponent(Tsr.class, g->{
            forComponent(Optimizer.class, o->((Optimizer)o).optimize());
            remove(Tsr.class);
            FunctionBuilder.build("I[0]<-(I[0]+I[1])", false).activate(new Tsr[]{this, (Tsr)g});
        });
    }

    //TENSOR OPERATION (OVERLOADABLE):
    //=================================
    public Tsr T() {//Transposed!
        StringBuilder operation = new StringBuilder();
        for (int i=rank()-1; i>=0; i--) operation.append(i).append((i == 0) ? "" : ", ");
        operation = new StringBuilder("[" + operation + "]:(I[0])");
        return new Tsr(this, operation.toString());
    }

    public Tsr plus(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "(I[0]+I[1])");
    }
    public Tsr plus(Double value) {
        return plus(new Tsr(this.shape(), value));
    }
    public Tsr minus(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "(I[0]-I[1])");
    }
    public Tsr negative(){
        return FunctionBuilder.build("(-1*I[0])", false).activate(new Tsr[]{this});
    }
    public Tsr multiply(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "(I[0]*I[1])");
    }
    public Tsr multiply(Double value) {
        return multiply(new Tsr(this.shape(), value));
    }
    public Tsr div(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "(I[0]/I[1])");
    }
    public Tsr div(Double value) {
        return div(new Tsr(this.shape(), value));
    }
    public Tsr mod(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "(I[0]%I[1])");
    }
    public Tsr power(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "(I[0]^I[1])");
    }
    public Tsr power(Double value){
        return power(new Tsr(this.shape(), value));
    }
    public Tsr xor(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "(I[0]^I[1])");
    }
    public Tsr xor(Double value) {
        return xor(new Tsr(this.shape(), value));
    }
    public Tsr dot(Tsr b){
        Tsr a = this;
        int[][] fitter = AbstractNDArray.Utility.Indexing.makeFit(a.shape(), b.shape());
        boolean doReshape = false;
        for(int i=0; i<fitter[0].length && !doReshape; i++) if(fitter[0][i]!=i) doReshape = true;
        for(int i=0; i<fitter[1].length && !doReshape; i++) if(fitter[1][i]!=i) doReshape = true;
        if(doReshape){
            a = Function.create(AbstractNDArray.Utility.Stringify.strConf(fitter[0])+":(I[0])").activate(a);
            b = Function.create(AbstractNDArray.Utility.Stringify.strConf(fitter[0])+":(I[0])").activate(b);
        }
        Tsr result = Function.create("I[0]xI[1]").activate(new Tsr[]{a, b});
        return result;
    }

    public Tsr label(String[][] labels) {
        IndexAlias indexAlias = (IndexAlias)find(IndexAlias.class);
        if (indexAlias ==null) {
            indexAlias = new IndexAlias(this.rank());
            add(indexAlias);
        }
        for(int i=0; i<labels.length; i++) {
            if (labels[i]!=null) {
                for (int ii=0; ii<labels[i].length; ii++) {
                    if (labels[i][ii]!=null) indexAlias.set(i, labels[i][ii], ii);
                }
            }
        }
        return this;
    }

    public Tsr label(List<List<Object>> labels) {
        IndexAlias indexAlias = (IndexAlias)find(IndexAlias.class);
        if (indexAlias ==null) add(new IndexAlias(labels));
        return this;
    }

    public Tsr label(Map<Object, List<Object>> labels) {
        this.add(new IndexAlias(labels, this));
        return this;
    }

    public Tsr putAt(Object key, Tsr value) {
        if (value.isEmpty()) throw new IllegalArgumentException("Provided tensor is empty!");
        Tsr slice = (key==null) ? this : (Tsr)getAt(key);
        boolean valueIsDeviceVisitor = false;
        if (slice.isOutsourced() && !value.isOutsourced()){
            Device device = (Device)slice.find(Device.class);
            device.add(value);
            valueIsDeviceVisitor = true;
        }
        if (this.isEmpty() && slice.isEmpty() || slice.size()!=value.size()) _become(value);//Rethink this a little
        else new Tsr(new Tsr[]{slice, value}, "I[0]<-I[1]", false);
        if (valueIsDeviceVisitor) ((Device)value.find(Device.class)).get(value);
        return this;
    }

    public double getAt(int[] idx){
        return value64()[i_of_idx(idx)];
    }

    public Object getAt(Object key) {
        if (key==null) return this;
        if (key instanceof List) if (((List)key).isEmpty()) return this;
        int[] idxbase = null;
        int[] newShape = new int[this.rank()];
        if (key instanceof List) {
            key = ((List)key).toArray();
            boolean allInt = true;
            for(Object o : (Object[])key) allInt = allInt && o instanceof Integer;
            if (allInt) {
                key = _intArray((Object[]) key);
                idxbase = (int[])key;
                if(key != null) {
                    for(int i=0; i<this.rank(); i++) idxbase[i] = (idxbase[i]<0)?_shape[i]+idxbase[i]:idxbase[i];
                    return IO.getFrom(this, idxbase);
                }
            } else {
                boolean hasScale = false;
                for (Object o : (Object[])key) hasScale = hasScale || o instanceof Map;
                idxbase = new int[((hasScale)?2:1)*this.rank()];
                Object[] ranges = (Object[])key;
                _configureSubsetFromRanges(ranges, idxbase, newShape, 0);
            }
        }//...not simple slice... Advanced:
        else if (key instanceof Map)// ==> i, j, k slicing!
        {
            idxbase = new int[this.rank()*2];
            Object[] ranges = ((Map)key).keySet().toArray();
            _configureSubsetFromRanges(ranges, idxbase, newShape, 0);
            Object[] steps = ((Map)key).values().toArray();
            for (int i=rank(); i<2*this.rank(); i++){
                idxbase[i] = (Integer)steps[i-rank()];
                newShape[i-rank()] /= (Integer)steps[i-rank()];
            }
        }
        Tsr subset = new Tsr();
        subset._value = this._value;
        subset._translation = this._translation;
        subset._idxmap = _cached(Utility.Indexing.newTlnOf(newShape));
        subset._shape = _cached(newShape);

        int[] newSpread = new int[rank()];
        int[] newOffset = new int[rank()];
        Arrays.fill(newSpread, 1);
        if (idxbase.length==2*rank()){
            for(int i=rank(); i<idxbase.length; i++) idxbase[i] = (idxbase[i]==0)?1:idxbase[i];
        }
        for(int i=0; i<idxbase.length; i++){
            if(i>=rank()) newSpread[i-rank()] = idxbase[i];
            else newOffset[i] = idxbase[i];
        }
        subset._spread = newSpread;
        subset._offset = newOffset;

        if (this.isOutsourced()){
            Device device = (Device) this.find(Device.class);
            device.add(subset, this);
            subset.setIsOutsourced(true);
        }
        if (this.isVirtual()) subset.setIsVirtual(true);
        subset.add(new Relation().addParent(this));
        Relation parent = (Relation) find(Relation.class);
        parent = (parent!=null)?parent:new Relation();
        parent.addChild(subset);
        this.add(parent);
        return subset;
    }

    /**
     *
     * @param ranges Elements of this array might be multiple things:
     *               - A map whose first entry represents a mapping between range and steps.
     *               - A list from which a first and last entry will be interpreted as range.
     *               - Any other object which might bew found in a 'IndexAlias' component.
     * @param idxbase Start index for every rank.
     * @param newShape New shape of the new sub-tensor.
     * @param offset Rank offset incremented according to recursive calls.
     * @return A new rank index.
     */
    private int _configureSubsetFromRanges(Object[] ranges, int[] idxbase, int[] newShape, int offset){
        for (int i=0; i<ranges.length; i++) {
            int first = 0;
            int last = 0;
            if (!(ranges[i] instanceof  List)){
                if (ranges[i] instanceof Map){
                    Object[] ks = ((Map)ranges[i]).keySet().toArray();
                    Object[] steps = ((Map)ranges[i]).values().toArray();
                    int newI = _configureSubsetFromRanges(ks, idxbase, newShape, i+offset);
                    for (int ii=rank(); ii<(rank()+steps.length); ii++) {
                        idxbase[ii+i+offset] = (Integer)steps[ii-rank()];
                        newShape[ii+i+offset-rank()] /= idxbase[ii+i+offset];
                    }
                    i = newI;
                    continue;
                } else {
                    IndexAlias indexAlias = (IndexAlias)find(IndexAlias.class);
                    if (indexAlias !=null){
                        int position = indexAlias.get(ranges[i], i+offset);
                        first = position;
                        last = position;
                    } else {
                        throw new IllegalStateException("[Tsr]: Given indexAlias key at axis "+i+offset+" not found!");
                    }
                }
            }else{
                ranges[i] = ((List)ranges[i]).toArray();
                ranges[i] = (((Object[])ranges[i])[0] instanceof List)?((List)((Object[])ranges[i])[0]).toArray():((Object[])ranges[i]);
                if (!(((Object[])(ranges[i]))[0] instanceof Integer) || !(((Object[])(ranges[i]))[((Object[])(ranges[i])).length-1] instanceof Integer)){
                    IndexAlias indexAlias = (IndexAlias)find(IndexAlias.class);
                    if (!(((Object[])(ranges[i]))[0] instanceof Integer)){
                        if (indexAlias !=null){
                            first = indexAlias.get(((Object[])(ranges[i]))[0], i+offset);
                        }
                    }  else {
                        first = (Integer) ((Object[])(ranges[i]))[0];
                    }
                    if (!(((Object[])(ranges[i]))[((Object[])(ranges[i])).length-1] instanceof Integer)){
                        if (indexAlias !=null){
                            last = indexAlias.get(((Object[])(ranges[i]))[((Object[])(ranges[i])).length-1], i+offset);
                        }
                    } else {
                        last = (Integer) ((Object[])(ranges[i]))[((Object[])(ranges[i])).length-1];
                    }
                } else {
                    first = ((Integer)((Object[])ranges[i])[0]);
                    last = ((Integer)((Object[])ranges[i])[((Object[])ranges[i]).length-1]);
                }
            }
            if (first<0 && last<0 && first>last){
                int temp = first;
                first = last;
                last = temp;
            }
            first = (first < 0) ? _shape[i]+first : first;
            last = (last < 0) ? _shape[i]+last : last;
            newShape[i+offset] = (last - first) + 1;
            idxbase[i+offset] = first;
        }
        return ranges.length+offset-1;
    }
    
    public static class IO
    {
        private IO(){}

        public static double getFrom(Tsr t, int i) {
            if (t.isEmpty() || t.isUndefined()) return 0;
            else if (t.isVirtual()) return t.value64()[0];
            return t.value64()[t.i_of_i(i)];
        }

        public static double getFrom(Tsr t, int[] idx) {
            t.setIsVirtual(false);
            return t.value64()[t.i_of_idx(idx)];
        }

        public static void setInto(Tsr t, int i, double value) {
            t.setIsVirtual(false);
            t.value64()[t.i_of_i(i)] = value;
        }

        public static void setInto(Tsr t, int[] idx, double value) {
            t.setIsVirtual(false);
            t.value64()[t.i_of_idx(idx)] = value;
        }

        public static void addInto(Tsr t, int i, double value) {
            t.setIsVirtual(false);
            t.value64()[t.i_of_i(i)] += value;
        }

        public static void addInto(Tsr t, int[] idx, double value) {
            t.setIsVirtual(false);
            t.value64()[t.i_of_idx(idx)] += value;
        }

        public static Tsr addInto(Tsr t, Tsr source) {
            if (t.isVirtual() && source.isVirtual()) t.value64()[0] += source.value64()[0];
            else FunctionBuilder.build("I[0]<-(I[0]+I[1])", false).activate(new Tsr[]{t, source});
            return source;
        }

        public static void subInto(Tsr t, int i, double value) {
            t.setIsVirtual(false);
            t.value64()[t.i_of_i(i)] -= value;
        }

        public static void subInto(Tsr t, int[] idx, double value) {
            t.setIsVirtual(false);
            t.value64()[t.i_of_idx(idx)] -= value;
        }

        public static void subInto(Tsr t, Tsr source) {
            if (t.isVirtual() && source.isVirtual()) {
                t.value64()[0] -= source.value64()[0];
            } else {
                if (t.isVirtual()) t.setIsVirtual(false);
                int[] index = new int[t.shape().length];
                int size = t.size();
                for (int i = 0; i < size; i++) {
                    IO.subInto(t, index, IO.getFrom(source, index));
                    Utility.Indexing.increment(index, t.shape());
                }
            }
        }

        public static void mulInto(Tsr t, int i, double value) {
            t.setIsVirtual(false);
            t.value64()[t.i_of_i(i)] *= value;
        }

        public static void mulInto(Tsr t, int[] idx, double value) {
            t.setIsVirtual(false);
            t.value64()[t.i_of_idx(idx)] *= value;
        }

    }

    public static class Exec
    {
        private Exec(){}

        public static Tsr reshaped(Tsr tensor, int[] newForm, boolean newTsr) {
            tensor = (newTsr) ? new Tsr(tensor, true) : tensor;
            //tensor = (newTsr) ? tensor.getAt(null) : tensor;
            tensor._shape = _cached(Utility.Indexing.shpCheck(Utility.Indexing.rearrange(tensor._shape, newForm), tensor));
            tensor._translation = _cached(Utility.Indexing.rearrange(tensor._translation, tensor._shape, newForm));
            tensor._idxmap =  _cached(Utility.Indexing.newTlnOf(tensor._shape));
            int[] newShp = new int[newForm.length];
            for (int i = 0; i < newForm.length; i++) {
                if (newForm[i] < 0) newShp[i] = 1;
                else if (newForm[i] >= 0) newShp[i] = tensor._spread[newForm[i]];
            }
            tensor._spread =  _cached(newShp);
            newShp = new int[newForm.length];
            for (int i = 0; i < newForm.length; i++) {
                if (newForm[i] < 0) newShp[i] = 0;
                else if (newForm[i] >= 0) newShp[i] = tensor._offset[newForm[i]];
            }
            tensor._offset =  _cached(newShp);
            return tensor;
        }

    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    public Tsr setValue64(double[] value) {
        if (this.isOutsourced()) ((Device) this.find(Device.class)).overwrite64(this, value);
        else _value = value;
        return this;
    }

    public Tsr setValue32(float[] value) {
        if (this.isOutsourced()) ((Device) this.find(Device.class)).overwrite32(this, value);
        else _value = value;
        return this;
    }

    public Tsr setValue(Object value) {
        if (value instanceof float[]) this.setValue32((float[])value);
        else if(value instanceof  double[]) this.setValue64((double[])value);
        else if(value instanceof Float) {
            this.setIsVirtual(true);
            if(this.is32()) ((float[])_value)[0] = (Float) value;
            else ((double[])_value)[0] = ((Float)value).doubleValue();
        } else if (value instanceof Double) {
            this.setIsVirtual(true);
            if(this.is64()) ((double[])_value)[0] = (Double) value;
            else ((float[])_value)[0] = ((Double)value).floatValue();
        }
        return this;
    }

    public Object getValue(){
        if(this.isOutsourced()){
            Device device = ((Device)find(Device.class));
            return (this.is32())?device.value32Of(this):device.value64Of(this);
        }
        return _value;
    }

    public double[] gradient64() {
        Tsr gradient = (Tsr)this.find(Tsr.class);
        if(gradient==null) return new double[0];
        return (this.is32())? DataHelper.floatToDouble(gradient.value32()):gradient.value64();
    }

    public float[] gradient32(){
        Tsr gradient = (Tsr)this.find(Tsr.class);
        if(gradient==null) return new float[0];
        return (this.is64())?DataHelper.doubleToFloat(gradient.value64()): gradient.value32();
    }

    public Tsr addToGradient(Tsr error) {
        if(!forComponent(Tsr.class,  g ->
            this.add(FunctionBuilder.build("I[0]<-(I[0]+I[1])", false).activate(new Tsr[]{(Tsr)g, error}))
        )){
            this.add(error).forComponent(Device.class, d ->((Device)d).add(error));
        }
        return this;
    }

    public Tsr to32() {
        if (this.is64()){
            Device device = (Device) this.find(Device.class);
            if (device!=null) device.get(this);
            _value = DataHelper.doubleToFloat((double[])_value);
            forComponent(Tsr.class, g ->((Tsr)g).to32());
            if (device!=null) device.add(this);
        }
        return this;
    }

    public Tsr to64() {
        if (this.is32()) {
            Device device = (Device) this.find(Device.class);
            if (device!=null) device.get(this);
            _value = DataHelper.floatToDouble((float[])_value);
            forComponent(Tsr.class, g ->((Tsr)g).to64());
            if (device!=null) device.add(this);
        }
        return this;
    }

    public double value64(int i) {
        if (this.isVirtual()){
            if (this.is64()) return ((double[])_value)[0];
            else return ((float[])_value)[0];
        } else {
            if (this.is64()) return ((double[])_value)[i];
            else return ((float[])_value)[i];
        }
    }

    public double[] value64() {
        if (_value == null && this.isOutsourced() && this.has(Device.class)) {
            return ((Device) this.find(Device.class)).value64Of(this);
        }
        double[] newValue = (this.is64())?(double[])_value: DataHelper.floatToDouble((float[])_value);
        if (this.isVirtual() && newValue!=null) {
            newValue = new double[this.size()];
            double[] value = (this.is64())?(double[])_value:DataHelper.floatToDouble((float[])_value);
            Arrays.fill(newValue, value[0]);
        }
        return newValue;
    }

    public float value32(int i) {
        if (this.isVirtual()){
            if (this.is64()) return (float) ((double[])_value)[0];
            else return ((float[])_value)[0];
        } else {
            if (this.is64()) return (float) ((double[])_value)[i];
            else return ((float[])_value)[i];
        }
    }

    public float[] value32() {
        if (_value == null && this.isOutsourced() && this.has(Device.class)) {
            return ((Device) this.find(Device.class)).value32Of(this);
        }
        float[] newValue = (this.is64())?DataHelper.doubleToFloat((double[])_value):(float[])_value;
        if (this.isVirtual() && newValue!=null) {
            newValue = new float[this.size()];
            Arrays.fill(newValue, newValue[0]);
        }
        return newValue;
    }

    public Tsr setValue(double[] newValue) {
        _value = newValue;
        if (this.isOutsourced() && newValue != null) ((Device) this.find(Device.class)).add(this);
        return this;
    }



    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //DISPLAY :
    //=========================
    public String toString(String mode){
        return _toString(mode, (mode.contains("f"))?"    ":null);
    }

    protected String _toString(String mode, String deep) {
        String base = (deep==null)?"":"\n"+deep;
        String delimiter = (deep==null)?"":"    ";
        String half = (deep==null)?"":"  ";
        String deeper = (deep==null)?deep:deep+delimiter;
        int max = (mode.contains("s"))?3:50;
        if (this.isEmpty()) return "empty";
        else if (this.isUndefined()) return "undefined";
        StringBuilder strShape = new StringBuilder();
        int[] shape = shape();
        for (int i = 0; i < _shape.length; i++) {
            strShape.append(shape[i]);
            if (i < shape.length - 1) strShape.append("x");
        }
        boolean compact = mode.contains("c");
        strShape = new StringBuilder(
                (Neureka.instance().settings().view().legacy())
                        ? "[" + strShape + "]"
                        : "(" + strShape + ")"
        );
        if (mode.contains("shape") || mode.contains("shp")) return strShape.toString();
        String asString = "";
        asString += _stringified((value64()), compact, max);
        asString = strShape +
                (
                        (Neureka.instance().settings().view().legacy())
                                ? ":(" + asString + ")"
                                : ":[" + asString + "]"
                );
        if (mode.contains("g") && this.rqsGradient()) {
            asString += ":g:";
            Tsr gradient = (Tsr) this.find(Tsr.class);
            if (gradient!=null) asString += gradient.toString("c").replace(strShape+":","");
            else asString+="(null)";
        }
        if (mode.contains("r") && this.has(GraphNode.class) && ((GraphNode) this.find(GraphNode.class)).size() > 0) {
            GraphNode node = (GraphNode) this.find(GraphNode.class);
            AtomicReference<String> enclosed = new AtomicReference<>("; ");
            node.forEachDerivative((t, d) -> {
                if (d.derivative()==null){
                    enclosed.set(enclosed.get() + "->d(null), ");
                } else {
                    enclosed.set(enclosed.get() +
                            base+"=>d|[ " +
                            base+delimiter+    d.derivative()._toString(mode, deeper) + " " +
                            base+half+"]|:t{ " +
                            base+delimiter+    ((t.getPayload()!=null)?t.getPayload()._toString(mode, deeper):t.toString("")) + " " +
                            base+half+"}, ");
                }
            });
            asString += enclosed.get();
        }
        if (mode.contains("d") && this.has(GraphNode.class) && ((GraphNode) this.find(GraphNode.class)).size() > 0) {
            GraphNode node = (GraphNode) this.find(GraphNode.class);
            if (node.mode() != 0) {
                AtomicReference<String> asAR = new AtomicReference<>("; ");
                node.forEachDerivative((t, d) -> {
                    if (d.derivative()==null) asAR.set(asAR.get() + "->d(null), ");
                    else asAR.set(asAR.get() + "->d" + d.derivative()._toString(mode, deeper) + ", ");
                });
                asString += asAR.get();
            }
        }
        return asString;
    }

    private String _stringified(double[] v, boolean format, int max){
        StringBuilder asString = new StringBuilder();
        int size = this.size();
        int trim = (size-max);
        size = (trim > 0) ? max : size;
        for (int i = 0; i < size; i++) {
            String vStr;
            if (format) vStr = Utility.Stringify.formatFP(v[(this.isVirtual()) ? 0 : i_of_i(i)]);
            else vStr = String.valueOf(v[(this.isVirtual()) ? 0 : i_of_i(i)]);
            asString.append(vStr);
            if (i < size - 1) asString.append(", ");
            else if (trim > 0) asString.append(", ... + ").append(trim).append(" more");
        }
        return asString.toString();
    }

    public String toString() {
        return toString("dgc");
    }


    public static void makeFit(Tsr[] tsrs){
        int largest = 0;
        for (Tsr t : tsrs) if (t.rank()>largest) largest = t.rank();
        for (int i=0; i<tsrs.length; i++) {
            if (tsrs[i].rank()!=largest) {
                int[] oldShape = tsrs[i].shape();
                int[] newReshape = new int[largest];
                int padding = largest-oldShape.length;
                for (int ii=0; ii<padding; ii++) newReshape[ii] = -1;
                for (int ii=padding; ii<largest; ii++) newReshape[ii] = i-padding;
                Function f = Function.create(
                    AbstractNDArray.Utility.Stringify.strConf(newReshape) +":(I[0])"
                );
                tsrs[i] = f.activate(tsrs[i]);
            }
        }

    }

    public static class Create
    {
        private Create(){}

        public  static Tsr E(int[] shape){
            return new Tsr(shape, 2.7182818284590452353602874713527);
        }

        public static Tsr newRandom(int[] shape){
            return newRandom(shape, 8701252152903546L);
        }

        public static Tsr newRandom(int[] shape, long seed){
            int size = Utility.Indexing.szeOfShp(shape);
            return new Tsr(shape, DataHelper.newSeededDoubleArray(seed, size));
        }

        public static Tsr newTsrLike(Tsr template, double value) {
            Tsr t = _newEmptyLike(template);
            if (template.is32()) t.setValue((float)value);
            else t.setValue(value);
            if (template.isOutsourced()) ((Device)template.find(Device.class)).add(t);
            return t;
        }

        public static Tsr newTsrLike(Tsr template) {//The output tensor will not have gradients!
            Tsr t = _newEmptyLike(template);
            if (template.is32()) t.setValue32(new float[template.size()]);
            else t.setValue64(new double[template.size()]);
            if (template.isOutsourced()) ((Device)template.find(Device.class)).add(t);
            return t;
        }

        private static Tsr _newEmptyLike(Tsr template) {
            Tsr t = new Tsr();
            t._shape = template._shape;
            t._idxmap = template._idxmap;
            t._translation = template.translation();
            t._spread = template.spread();
            t._offset = template.offset();
            return t;
        }

    }

}
