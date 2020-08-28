package neureka;

import groovy.lang.IntRange;
import neureka.calculus.environment.ExecutionCall;
import neureka.ndim.AbstractNDArray;
import neureka.acceleration.host.HostCPU;
import neureka.acceleration.Device;
import neureka.framing.IndexAlias;
import neureka.framing.Relation;
import neureka.calculus.Function;
import neureka.calculus.factory.assembly.FunctionBuilder;
import neureka.autograd.GraphNode;
import neureka.autograd.JITProp;
import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.virtual.VirtualNDConfiguration;
import neureka.optimization.Optimizer;
import neureka.utility.DataHelper;

import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

public class Tsr extends AbstractNDArray<Tsr> implements Component<Tsr>
{
    static { _CPU = HostCPU.instance(); }

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
    private static final int GRADIENT_APPLY_RQD_MASK = 8;

    /**
     *  The version of the data stored value within this tensor.
     *  Gets incremented every time an inline operation occurs!
     */
    private int _version = 0;

    public int version() {
        return _version;
    }

    public Tsr incrementVersionBecauseOf( ExecutionCall call ){
        if ( Neureka.instance().settings().autograd().isPreventingInlineOperations() ) {
            _version ++;
            GraphNode node = find( GraphNode.class );
            if ( node != null && node.referenceVersion() != this._version ) {
                if ( node.usesAD() || node.isUsedAsDerivative() ) {
                    throw new IllegalStateException(
                            "Inline operation occurred on tensor which is part of a computation graph node with autograd support!\n" +
                            "The following OperationType caused an internal version mismatch: '"+call.getType().getFunction()+"'"
                    );
                }
            }
        }
        return this;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public Tsr setRqsGradient( boolean rqsGradient ) {
        if ( rqsGradient() != rqsGradient && !rqsGradient ) this.remove(Tsr.class);
        _setRqsGradient(rqsGradient);
        return this;
    }

    public boolean rqsGradient() {
        return (_flags & RQS_GRADIENT_MASK) == RQS_GRADIENT_MASK;
    }

    protected void _setRqsGradient( boolean rqsGradient ) {
        if ( rqsGradient() != rqsGradient ) {
            if ( rqsGradient ) _flags += RQS_GRADIENT_MASK;
            else _flags -= RQS_GRADIENT_MASK;
        }
    }

    //------------------------------------------------------------------------------------------------------------------

    public Tsr setIsOutsourced( boolean isOutsourced ) {
        _setIsOutsourced( isOutsourced );
        if ( isOutsourced ) {
            _value = null;
        } else if (
                !forComponent(
                    Device.class,
                    d -> {
                        if ( d.has(this) ) d.get( this );
                        this.remove( Device.class );
                        forComponent(
                            Tsr.class,
                            gradient ->
                            gradient.forComponent( Device.class, gd -> {
                                if ( gd.has(gradient) ) gd.get(gradient);
                                gradient.remove(Device.class);
                            })
                        );
                    }
                ) && _value == null
        ){
            setIsVirtual( true );
        }
        return this;
    }

    public boolean isOutsourced() {
        return (_flags & IS_OUTSOURCED_MASK) == IS_OUTSOURCED_MASK;
    }

    protected void _setIsOutsourced( boolean isOutsourced ) {
        if ( isOutsourced() != isOutsourced ) {
            if ( isOutsourced ) _flags += IS_OUTSOURCED_MASK;
            else _flags -= IS_OUTSOURCED_MASK;
        }
    }

    //------------------------------------------------------------------------------------------------------------------

    public Tsr setIsVirtual( boolean isVirtual ) {
        if ( isVirtual() != isVirtual ) {
            Device device = this.find( Device.class );
            if ( device != null ) device.get( this );
            double v = ( _value == null ) ? 0 : ((this.is64())?((double[])_value)[0]:((float[])_value)[0]);
            if ( isVirtual ) {
                _value = new double[]{v};
                Relation parent = find( Relation.class );
                if ( parent!=null ) parent.foreachChild( c -> c._value=_value);
            } else {
                _value = ( this.is64() ) ? new double[this.size()] : new float[this.size()];
                int length = ( this.is64() ) ? ((double[])_value).length : ((float[])_value).length;
                for ( int i = 0; i < length; i++ ) {
                    if ( this.is64() ) ((double[])_value)[i] = v;
                    else ((float[])_value)[i] = (float)v;
                }
            }
            _setIsVirtual( isVirtual );
            if( _conf!=null ) _configureFromNewShape( _conf.shape(), isVirtual );
            if( device!=null ) device.add(this);
        } else if (isVirtual && _value==null) _value = new double[]{0};
        return this;
    }

    public boolean isVirtual() {
        return (_flags & IS_VIRTUAL_MASK) == IS_VIRTUAL_MASK;
    }

    protected void _setIsVirtual( boolean isVirtual ) {
        if ( isVirtual() != isVirtual ) {
            if ( isVirtual ) _flags += IS_VIRTUAL_MASK;
            else _flags -= IS_VIRTUAL_MASK;
        }
    }

    //------------------------------------------------------------------------------------------------------------------

    public Tsr setGradientApplyRqd( boolean applyRequested ) {
        if ( gradientApplyRqd() != applyRequested ) {
            if ( applyRequested ) _flags += GRADIENT_APPLY_RQD_MASK;
            else _flags -= GRADIENT_APPLY_RQD_MASK;
        }
        return this;
    }

    public boolean gradientApplyRqd() {
        return (_flags & GRADIENT_APPLY_RQD_MASK) == GRADIENT_APPLY_RQD_MASK;
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
    protected < T extends Component<Tsr> > T _addOrReject( T newComponent )
    {
        if (newComponent instanceof Device && !((Device)newComponent).has(this))
        {
            if ( this.has( Relation.class ) ) {
                Relation relation = find( Relation.class );
                if ( relation.hasParent() ) { // Root needs to be found ! :
                    Tsr root = relation.findRootTensor();
                    ((Device)newComponent).add(root);
                    root.find( Relation.class ).foreachChild( c -> c.setIsOutsourced(true) );
                } else { // This is root ! :
                    relation.foreachChild( c -> c.setIsOutsourced(true) );
                    ((Device)newComponent).add(this);
                }
            } else {
                ((Device)newComponent).add(this);
            }
            if ( ((Device)newComponent).has(this) ) setIsOutsourced(true);
        } else if (newComponent instanceof Tsr) {
            if(
                    ((Tsr)newComponent).shape().hashCode()!=this.shape().hashCode() ||
                    Arrays.hashCode(((Tsr)newComponent).getNDConf().shape()) != Arrays.hashCode(_conf.shape())
            ) newComponent = null;
        }
        return newComponent;
    }

    /**
     * This method is executed when a is being removed from the tensor.
     * The public remove method is implemented in the super class
     * 'AbstractComponentOwner' from which this class inherits.
     * In this super class the component logic is implemented.
     *
     * @param newComponent A component used to access features. (GraphNode, IndexAlias, Relation, int[], ...)
     * @return The unchanged object or when rejected: null (component rejected)
     */
    @Override
    protected <T extends Component<Tsr>> T _removeOrReject(T newComponent)
    {
        if ( newComponent instanceof Device ) {

        }
        return newComponent;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // HIGH LEVEL PROPERTIES :

    public boolean isEmpty() {
        return _value == null && !this.isOutsourced();
    }

    public boolean isUndefined() {
        return _conf==null || _conf.shape() == null;
    }

    public boolean isSlice() {
        Relation child = find(Relation.class);
        return (child != null && child.hasParent());
    }

    public int sliceCount() {
        Relation child = find(Relation.class);
        return ( child!=null ) ? child.childCount() : 0;
    }

    public boolean isSliceParent(){
        Relation parent = find(Relation.class);
        return (parent!=null && parent.hasChildren());
    }

    public boolean belongsToGraph() {
        return this.has(GraphNode.class);
    }

    public boolean isLeave() {
        return (!this.has(GraphNode.class)) || this.find(GraphNode.class).isLeave();
    }

    public boolean isBranch() {
        return !this.isLeave();
    }

    public boolean hasGradient() {
        return this.has( Tsr.class );
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Direct Access to component (Device)

    /**
     * @return The device on which this tensor is stored or 'CPU' if it is not outsourced.
     */
    public Device device() {
        if (this.isOutsourced()) return this.find(Device.class);
        return _CPU;
    }

    /**
     *
     * @return The graph node of the computation graph to which this tensor belongs or null if not part of a graph.
     */
    public GraphNode graphNode(){
        return find(GraphNode.class);
    }

    /**
     *
     * @return Custom IndexAlias object.
     */
    public IndexAlias index(){
        return find(IndexAlias.class);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    protected Tsr _become( Tsr tensor ) {
        if ( tensor == null ) return this;
        _value = tensor._value;
        _conf = tensor._conf;
        _components = Collections.synchronizedList(new ArrayList<>());
        _flags = tensor._flags;
        if ( tensor._components != null ) { // Inform components about their new owner:
            _components.addAll( tensor._components );
            List<Component<Tsr>> snapshot = new ArrayList<>( tensor._components );
            for ( Component<Tsr> o : snapshot ) o.update( tensor, this );
        }
        tensor._value = null;
        tensor._conf = null;
        tensor._components = null;
        tensor._flags = -1;
        return this;
    }

    public Tsr delete() {
        forComponent(GraphNode.class, n -> {
            if ( n.isUsedAsDerivative() ) {
                throw new IllegalStateException("Cannot delete a tensor which is used as derivative by the AD computation graph!");
            }
        });
        forComponent( Device.class, d -> d.rmv( this ) );
        _flags = -1;
        _value = null;
        _conf = null;
        forComponent( Tsr.class, Tsr::delete );
        _components = null;
        return this;
    }


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /**
     * @param newShape
     */
    protected void _configureFromNewShape(int[] newShape, boolean makeVirtual) {
        int size = Utility.Indexing.szeOfShp(newShape);
        _value = (_value==null) ? new double[size] : _value;
        int length = (this.is64())?((double[])_value).length:((float[])_value).length;
        if (size != length && (!this.isVirtual() || !makeVirtual)) {
            throw new IllegalArgumentException("Size of shape does not match stored value64!");
        }
        if(makeVirtual) {
            _conf = VirtualNDConfiguration.construct(newShape);
        } else {
            int[] newTranslation = Utility.Indexing.newTlnOf(newShape);
            int[] newIdxmap = newTranslation;
            int[] newSpread = new int[newShape.length];
            Arrays.fill(newSpread, 1);
            int[] newOffset = new int[newShape.length];
            _conf = AbstractNDC.construct(newShape, newTranslation, newIdxmap, newSpread, newOffset);
        }
    }


    //CONSTRUCTION :
    //=========================

    public Tsr(){}

    //Generic construction: (Groovy, Scala, ...)
    public Tsr(Object arg){
        _construct(new Object[]{arg});
    }

    public Tsr(String equation, List<Object> inputs){
        _construct(
                inputs.stream().map(Tsr::new).toArray(Tsr[]::new),
                equation,
                true
        );
    }

    public Tsr(List arg1, String arg2) {
        if ((arg1).get(0) instanceof Integer) {
            List<Integer> shape = arg1;
            int[] shp = new int[shape.size()];
            for (int i=0; i<shp.length; i++) shp[i] = shape.get(i);
            _construct(shp, arg2);
        } else if ((arg1).get(0) instanceof Tsr) {
            _construct(((List<Tsr>)arg1).toArray(new Tsr[0]), arg2, true);
        } else {
            _construct(
                    ((List<Object>)arg1).stream().map(Tsr::new).toArray(Tsr[]::new),
                    arg2,
                    true
            );
        }
    }

    public Tsr(List<Integer> shape, List range){
        int[] shp = new int[shape.size()];
        for(int i=0; i<shp.length; i++) shp[i] = shape.get(i);
        double[] value = new double[Utility.Indexing.szeOfShp(shp)];
        if(range.size()==1 && range.get(0) instanceof IntRange) range = (List) range.get(0);
        for(int i=0; i<value.length; i++){
            if(range.get(i%range.size()) instanceof BigDecimal){
                value[i] = ((BigDecimal)range.get(i%range.size())).doubleValue();
            } else if (range.get(i%range.size()) instanceof Integer){
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
        if (args.length==1){
            if (args[0] instanceof Object[]){
                _construct((Object[]) args[0]);
                return;
            } else if (args[0] instanceof BigDecimal) {
                _construct(new int[]{1}, ((BigDecimal)args[0]).doubleValue());
                return;
            } else if(args[0] instanceof Integer) {
                _construct(new int[]{1}, ((Integer)args[0]).doubleValue());
                return;
            } else {
                throw new IllegalArgumentException(
                        "Cannot create tensor from argument of type '"+args[0].getClass().getName()+"'!"
                );
            }
        }
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
        ArrayList<Tsr> tsrList = new ArrayList<>();
        for (Object o : args) {
            containsString = (o instanceof String) || containsString;
            if (o instanceof Tsr) {
                tsrList.add( (Tsr)o );
                numberOfTensors++;
            }
        }
        boolean doAD = true;
        Tsr[] tsrs = new Tsr[numberOfTensors];
        StringBuilder f = new StringBuilder();
        int ti=0;
        for (Object o : args) {
            if (tsrList.contains(o)){
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
        this._configureFromNewShape(shape, false);
    }

    public Tsr(int[] shape, double value) {
        _construct(shape, value);
    }

    private void _construct(int[] shape, double value){
        int size = Utility.Indexing.szeOfShp(shape);
        _value = new double[1];
        this.setIsVirtual( size > 1 );
        this._configureFromNewShape(shape, size > 1);
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
        this._configureFromNewShape(shape, false);
    }

    // TRACKED COMPUTATION :
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
        if (!forComponent(GraphNode.class, node -> node.backward(error)) && this.rqsGradient()) {
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
        backward(new Tsr(_conf.shape(), value));
        return this;
    }

    public void applyGradient() {
        forComponent(JITProp.class, JITProp::execute);
        forComponent(Tsr.class, g -> {
            forComponent(Optimizer.class, o -> o.optimize(this));
            remove(Tsr.class);
            boolean inlineSafety = Neureka.instance().settings().autograd().isPreventingInlineOperations();
            if ( inlineSafety ) Neureka.instance().settings().autograd().setIsPreventingInlineOperations(false);
            Function.Detached.PLUS_ASSIGN.call(new Tsr[]{this, g});
            if ( inlineSafety ) Neureka.instance().settings().autograd().setIsPreventingInlineOperations(true);
        });
    }

    //TENSOR OPERATION (OVERLOADABLE):
    //=================================
    public Tsr T() {//Transposed!
        StringBuilder operation = new StringBuilder();
        for ( int i = rank()-1; i >= 0; i-- ) operation.append(i).append((i == 0) ? "" : ", ");
        operation = new StringBuilder("[" + operation + "]:(I[0])");
        return new Tsr(this, operation.toString());
    }

    public Tsr plus(Tsr other) {
        return Function.PLUS.call(new Tsr[]{this, other});
    }
    public Tsr plusAssign(Tsr other){
        return Function.Detached.PLUS_ASSIGN.call(new Tsr[]{this, other});
    }
    public Tsr plus(Double value) {
        return plus(new Tsr(this.shape(), value));
    }
    public Tsr minus(Tsr other) {
        return Function.MINUS.call(new Tsr[]{this, other});
    }
    public Tsr minusAssign(Tsr other){
        return Function.Detached.MINUS_ASSIGN.call(new Tsr[]{this, other});
    }
    public Tsr negative(){
        return Function.NEG.call(new Tsr[]{this});
    }
    public Tsr multiply(Tsr other) {
        return Function.MUL.call(new Tsr[]{this, other});
    }
    public Tsr timesAssign(Tsr other){
        return Function.Detached.MUL_ASSIGN.call(new Tsr[]{this, other});
    }
    public Tsr multiply(Double value) {
        return multiply(new Tsr(this.shape(), value));
    }
    public Tsr div(Tsr other) {
        return Function.DIV.call(new Tsr[]{this, other});
    }
    public Tsr div(Double value) {
        return div(new Tsr(this.shape(), value));
    }
    public Tsr divAssign(Tsr other){
        return Function.Detached.DIV_ASSIGN.call(new Tsr[]{this, other});
    }
    public Tsr mod(Tsr other) {
        return Function.MOD.call(new Tsr[]{this, other});
    }
    public Tsr modAssign(Tsr other){
        return Function.Detached.MOD_ASSIGN.call(new Tsr[]{this, other});
    }
    public Tsr power(Tsr other) {
        return Function.POW.call(new Tsr[]{this, other});
    }
    public Tsr power(Double value){
        return power(new Tsr(this.shape(), value));
    }
    public Tsr xor(Tsr other) {
        return Function.POW.call(new Tsr[]{this, other});
    }
    public Tsr xor(Double value) {
        return xor(new Tsr(this.shape(), value));
    }
    public Tsr dot(Tsr b){
        Tsr a = this;
        int[][] fitter = AbstractNDArray.Utility.Indexing.makeFit(a.getNDConf().shape(), b.getNDConf().shape());
        boolean doReshape = false;
        for(int i=0; i<fitter[0].length && !doReshape; i++) if(fitter[0][i]!=i) doReshape = true;
        for(int i=0; i<fitter[1].length && !doReshape; i++) if(fitter[1][i]!=i) doReshape = true;
        if(doReshape){
            a = Function.create(AbstractNDArray.Utility.Stringify.strConf(fitter[0])+":(I[0])").call(a);
            b = Function.create(AbstractNDArray.Utility.Stringify.strConf(fitter[1])+":(I[0])").call(b);
        }
        return Function.X.call(new Tsr[]{a, b});
    }
    public boolean isCase(Tsr t){
        boolean[] found = {false};
        this.forComponent(Relation.class, r -> r.foreachChild( c -> {
                if (c.equals(t)) found[0]=true;
            }));
        return found[0];
    }
    public boolean contains(Tsr t){
        return isCase(t);
    }

    public Tsr label(String[][] labels) {
        IndexAlias indexAlias = find(IndexAlias.class);
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
        IndexAlias indexAlias = find(IndexAlias.class);
        if (indexAlias ==null) add(new IndexAlias(labels));
        return this;
    }

    public Tsr label(Map<Object, List<Object>> labels) {
        this.add(new IndexAlias(labels, this));
        return this;
    }

    public Tsr putAt(Object key, Tsr value) {
        if (value.isEmpty()) throw new IllegalArgumentException("Provided tensor is empty!");
        Tsr slice = ( key==null ) ? this : (Tsr)getAt(key);
        boolean valueIsDeviceVisitor = false;
        if (slice.isOutsourced() && !value.isOutsourced()){
            Device device = slice.find(Device.class);
            device.add(value);
            valueIsDeviceVisitor = true;
        }
        if (this.isEmpty() && slice.isEmpty() || slice.size()!=value.size()) _become(value);//Rethink this a little
        else new Tsr(new Tsr[]{slice, value}, "I[0]<-I[1]", false);
        if (valueIsDeviceVisitor) value.find(Device.class).get(value);
        return this;
    }

    public double getAt(int[] idx){
        return value64()[i_of_idx(idx)];
    }

    public Object getAt(Object i1, Object i2) {
        List<Object> args = Arrays.asList(i1, i2);
        return getAt(args);
    }

    public Object getAt(Object key) {
        if (key==null) return this;
        if(key instanceof Object[] && ((Object[])key).length==0) key = new ArrayList();
        if (key instanceof List && ((List)key).isEmpty()){
            if(this.isEmpty() || this.isUndefined()) return this;
            for(int e : this.shape()) {
                List<Integer> rangeAsList = new ArrayList<>();
                for(int i=0; i<e; i++) rangeAsList.add(i);
                ((List<Object>)key).add(rangeAsList);
            }
        } else if (key instanceof Integer) {
            key = Arrays.asList(_conf.idx_of_i((Integer)key));
        } else if (key instanceof Double) {
            key = Arrays.asList(_conf.idx_of_i((int)Math.floor((Double)key)));
        } else if (key instanceof BigDecimal) {
            key = Arrays.asList(_conf.idx_of_i(((BigDecimal)key).intValue()));
        }
        int[] idxbase = null;
        int[] newShape = new int[this.rank()];
        if (key instanceof List || key instanceof Object[]) {
            if (key instanceof  List) key = ((List)key).toArray();
            boolean allInt = true;
            for(Object o : (Object[])key) allInt = allInt && o instanceof Integer;
            if (allInt && ((Object[])key).length==rank()) {
                key = _intArray((Object[]) key);
                idxbase = (int[])key;
                if(key != null) {
                    for(int i=0; i<this.rank(); i++) idxbase[i] = (idxbase[i]<0)?_conf.shape(i)+idxbase[i]:idxbase[i];
                    return IO.getFrom(this, idxbase);
                }
            } else {
                boolean hasScale = false;
                for (Object o : (Object[])key) hasScale = hasScale || o instanceof Map;
                idxbase = new int[((hasScale)?2:1)*this.rank()];
                if (allInt) {
                    _configureSubsetFromRanges(new Object[]{_intArray((Object[]) key)}, idxbase, newShape, 0);
                } else {
                    _configureSubsetFromRanges((Object[])key, idxbase, newShape, 0);
                }

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
        int[] newTranslation = this._conf.translation();
        int[] newIdxmap = Utility.Indexing.newTlnOf(newShape);
        int[] newSpread = new int[rank()];
        int[] newOffset = new int[rank()];
        Arrays.fill(newSpread, 1);
        if ( idxbase.length == 2 * rank() ){
            for(int i=rank(); i<idxbase.length; i++) idxbase[i] = (idxbase[i]==0)?1:idxbase[i];
        }
        for(int i=0; i<idxbase.length; i++){
            if(i>=rank()) newSpread[i-rank()] = idxbase[i];
            else newOffset[i] = idxbase[i];
        }
        subset._conf = AbstractNDC.construct(newShape, newTranslation, newIdxmap, newSpread, newOffset);

        if (this.isOutsourced()){
            Device device = this.find(Device.class);
            device.add(subset, this);
            subset.setIsOutsourced(true);
        }
        if (this.isVirtual()) subset.setIsVirtual(true);
        subset.add(new Relation().addParent(this));
        Relation parent = find(Relation.class);
        parent = (parent!=null) ? parent : new Relation();
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
            if(ranges[i] instanceof int[]){
                if(ranges[i] instanceof int[]){
                    List<Integer> intList = new ArrayList<>(((int[])ranges[i]).length);
                    for (int ii : (int[])ranges[i]) intList.add(ii);
                    ranges[i] = intList;
                }
            } else if (ranges[i] instanceof String[]){
                if (ranges[i] instanceof String[]){
                    List<String> strList = new ArrayList<>(((String[])ranges[i]).length);
                    for (String ii : (String[])ranges[i]) strList.add(ii);
                    ranges[i] = strList;
                }
            }
            if (!(ranges[i] instanceof  List)) {
                if (ranges[i] instanceof Map) {
                    Object[] ks = ((Map) ranges[i]).keySet().toArray();
                    Object[] steps = ((Map) ranges[i]).values().toArray();
                    int newI = _configureSubsetFromRanges(ks, idxbase, newShape, i + offset);
                    for (int ii = rank(); ii < (rank() + steps.length); ii++) {
                        idxbase[ii + i + offset] = (Integer) steps[ii - rank()];
                        newShape[ii + i + offset - rank()] /= idxbase[ii + i + offset];
                    }
                    i = newI;
                    continue;
                } else if(ranges[i] instanceof Integer) {
                    first = (Integer) ranges[i];
                    last = (Integer) ranges[i];
                } else {
                    IndexAlias indexAlias = find(IndexAlias.class);
                    if (indexAlias !=null){
                        int position = indexAlias.get(ranges[i], i+offset);
                        first = position;
                        last = position;
                    } else {
                        throw new IllegalStateException("Given indexAlias key at axis "+(i+offset)+" not found!");
                    }
                }
            } else {
                ranges[i] = ((List)ranges[i]).toArray();
                ranges[i] = (((Object[])ranges[i])[0] instanceof List)?((List)((Object[])ranges[i])[0]).toArray():((Object[])ranges[i]);
                if (!(((Object[])(ranges[i]))[0] instanceof Integer) || !(((Object[])(ranges[i]))[((Object[])(ranges[i])).length-1] instanceof Integer)){
                    IndexAlias indexAlias = find(IndexAlias.class);
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
            first = (first < 0) ? _conf.shape(i)+first : first;
            last = (last < 0) ? _conf.shape(i)+last : last;
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
            else FunctionBuilder.build("I[0]<-(I[0]+I[1])", false).call(new Tsr[]{t, source});
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
                int[] index = new int[t.getNDConf().shape().length];
                int size = t.size();
                for (int i = 0; i < size; i++) {
                    IO.subInto(t, index, IO.getFrom(source, index));
                    Utility.Indexing.increment(index, t.getNDConf().shape());
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

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    public Tsr setValue64(double[] value) {
        if (this.isOutsourced()) this.find(Device.class).overwrite64(this, value);
        else if ( _value==null ) _value = value;
        else if ( _value instanceof float[] ) for ( int i=0; i<value.length; i++ ) ((float[])_value)[i] = (float) value[i];
        else if ( _value instanceof double[] ) for ( int i=0; i<value.length; i++ ) ((double[])_value)[i] = value[i];
        return this;
    }

    public Tsr setValue32(float[] value) {
        if (this.isOutsourced()) this.find(Device.class).overwrite32(this, value);
        else if ( _value==null ) _value = value;
        else if ( _value instanceof float[] ) for ( int i=0; i<value.length; i++ ) ((float[])_value)[i] = value[i];
        else if ( _value instanceof double[] ) for ( int i=0; i<value.length; i++ ) ((double[])_value)[i] = value[i];
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
            Device device = find(Device.class);
            return (this.is32())?device.value32f(this):device.value64f(this);
        }
        return _value;
    }

    public double[] gradient64() {
        Tsr gradient = this.find(Tsr.class);
        if(gradient==null) return new double[0];
        return (this.is32())? DataHelper.floatToDouble(gradient.value32()):gradient.value64();
    }

    public float[] gradient32(){
        Tsr gradient = this.find(Tsr.class);
        if(gradient==null) return new float[0];
        return (this.is64())?DataHelper.doubleToFloat(gradient.value64()): gradient.value32();
    }

    public Tsr addToGradient(Tsr error) {
        if(!forComponent(Tsr.class,  g ->
            this.add(Function.Detached.PLUS_ASSIGN.call(new Tsr[]{g, error}))
        )){
            this.add(error).forComponent(Device.class, d -> d.add(error));
        }
        return this;
    }

    public Tsr to32() {
        if (this.is64()){
            Device device = this.find(Device.class);
            if (device!=null) device.get(this);
            _value = DataHelper.doubleToFloat((double[])_value);
            forComponent(Tsr.class, Tsr::to32);
            if (device!=null) device.add(this);
        }
        return this;
    }

    public Tsr to64() {
        if (this.is32()) {
            Device device = this.find(Device.class);
            if (device!=null) device.get(this);
            _value = DataHelper.floatToDouble((float[])_value);
            forComponent(Tsr.class, Tsr::to64);
            if (device!=null) device.add(this);
        }
        return this;
    }

    // TODO: WRITE A UNIT TEST ABOUT THIS!!!
    public double value64(int i) {
        if (this.isVirtual()){
            if ( this.isOutsourced() ) {
                Device device = find(Device.class);
                return device.value64f(this, i);
            } else {
                if (this.is64()) return ((double[]) _value)[0];
                else return ((float[]) _value)[0];
            }
        } else {
            if ( this.isOutsourced() ) {
                Device device = find(Device.class);
                return device.value64f(this, i);
            } else {
                if (this.is64()) return ((double[])_value)[i];
                else return ((float[])_value)[i];
            }
        }
    }

    public double[] value64() {
        if (_value == null && this.isOutsourced() && this.has(Device.class)) {
            return this.find(Device.class).value64f(this);
        }
        double[] newValue = (this.is64())?(double[])_value: DataHelper.floatToDouble((float[])_value);
        if (this.isVirtual() && newValue!=null && this.size()>1) {
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
            return this.find(Device.class).value32f(this);
        }
        float[] newValue = (this.is64())?DataHelper.doubleToFloat((double[])_value):(float[])_value;
        if (this.isVirtual() && newValue!=null) {
            newValue = new float[this.size()];
            Arrays.fill(newValue, newValue[0]);
        }
        return newValue;
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
        int[] shape = _conf.shape();
        for (int i = 0; i < shape.length; i++) {
            strShape.append(shape[i]);
            if (i < shape.length - 1) strShape.append("x");
        }
        boolean compact = mode.contains("c");
        strShape = new StringBuilder(
                (Neureka.instance().settings().view().isUsingLegacyView())
                        ? "[" + strShape + "]"
                        : "(" + strShape + ")"
        );
        if (mode.contains("shape") || mode.contains("shp")) return strShape.toString();
        String asString = "";
        asString += _stringified((value64()), compact, max);
        asString = strShape +
                (
                        (Neureka.instance().settings().view().isUsingLegacyView())
                                ? ":(" + asString + ")"
                                : ":[" + asString + "]"
                );
        if ( mode.contains("g") && (this.rqsGradient() || this.hasGradient()) ) {
            asString += ":g:";
            Tsr gradient = this.find(Tsr.class);
            if (gradient!=null) asString += gradient.toString("c").replace(strShape+":","");
            else asString+= ((Neureka.instance().settings().view().isUsingLegacyView())?"(null)":"[null]");
        }
        if (mode.contains("r") && this.has(GraphNode.class) && this.find(GraphNode.class).size() > 0) {
            GraphNode node = this.find(GraphNode.class);
            AtomicReference<String> enclosed = new AtomicReference<>("; ");
            node.forEachDerivative((t, agent) -> {
                if (agent.derivative()==null){
                    enclosed.set(enclosed.get() + "->d(null), ");
                } else {
                    enclosed.set(enclosed.get() +
                            base+"=>d|[ " +
                            base+delimiter+    agent.derivative()._toString(mode, deeper) + " " +
                            base+half+"]|:t{ " +
                            base+delimiter+    ((t.getPayload()!=null)?t.getPayload()._toString(mode, deeper):t.toString("")) + " " +
                            base+half+"}, ");
                }
            });
            asString += enclosed.get();
        }
        if (mode.contains("d") && this.has(GraphNode.class) && this.find(GraphNode.class).size() > 0) {
            GraphNode node = this.find(GraphNode.class);
            if (node.mode() != 0) {
                AtomicReference<String> asAR = new AtomicReference<>("; ");
                node.forEachDerivative((t, agent) -> {
                    if (agent.derivative()==null) asAR.set(asAR.get() + "->d(null), ");
                    else asAR.set(asAR.get() + "->d" + agent.derivative()._toString(mode, deeper) + ", ");
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
                int[] oldShape = tsrs[i].getNDConf().shape();
                int[] newReshape = new int[largest];
                int padding = largest-oldShape.length;
                for (int ii=0; ii<padding; ii++) newReshape[ii] = -1;
                for (int ii=padding; ii<largest; ii++) newReshape[ii] = i-padding;
                Function f = Function.create(
                    AbstractNDArray.Utility.Stringify.strConf(newReshape) +":(I[0])"
                );
                tsrs[i] = f.call(tsrs[i]);
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
            if (template.isOutsourced()) template.find(Device.class).add(t);
            return t;
        }

        public static Tsr newTsrLike(Tsr template) {//The output tensor will not have gradients!
            Tsr t = _newEmptyLike(template);
            if (template.is32()) t.setValue32(new float[template.size()]);
            else t.setValue64(new double[template.size()]);
            if (template.isOutsourced()) template.find(Device.class).add(t);
            return t;
        }

        private static Tsr _newEmptyLike(Tsr template) {
            Tsr t = new Tsr();
            t._conf = template._conf;
            return t;
        }

    }



    @Override
    public void update(Tsr oldOwner, Tsr newOwner) {

    }


}
