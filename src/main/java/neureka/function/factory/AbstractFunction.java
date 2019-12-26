package neureka.function.factory;

import neureka.Tsr;
import neureka.acceleration.CPU;
import neureka.acceleration.Device;
import neureka.function.Function;
import neureka.function.factory.assembly.FunctionBuilder;
import neureka.function.factory.autograd.GraphNode;
import neureka.function.factory.implementations.FConstant;
import org.jetbrains.annotations.Contract;

import java.util.ArrayList;
import java.util.function.Supplier;

public abstract class AbstractFunction implements Function
{
    protected int _id;
    protected boolean _isFlat;
    protected boolean _doAD;
    protected ArrayList<Function> _src;

    /**
     * @param f_id
     * @param isFlat
     * @param source
     * @param doAD
     */
    protected AbstractFunction(int f_id, boolean isFlat, ArrayList<Function> source, boolean doAD) {
        _id = f_id;
        _isFlat = isFlat;
        _src = source;
        _doAD = doAD;
    }

    @Override
    public Function newBuild(String expression) {
        return FunctionBuilder.build(expression, true);
    }

    @Override
    public boolean isFlat() {
        return _isFlat;
    }

    @Override
    public int id() {
        return _id;
    }

    @Override
    public String type() {
        return TYPES.REGISTER[_id];
    }

    @Override
    public String toString() {
        String reconstructed = "";
        if (_src.size() == 1 && TYPES.REGISTER[_id].length() > 1) {
            String expression = _src.get(0).toString();
            if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                return TYPES.REGISTER[_id] + expression;
            }
            return TYPES.REGISTER[_id] + "(" + expression + ")";
        } else {
            reconstructed = ((TYPES.REGISTER[_id].equals(",")) ? "[" : "") + reconstructed;
            for (int i = 0; i < _src.size(); ++i) {
                if (_src.get(i) != null) {
                    if ((TYPES.REGISTER[_id].equals(","))) {
                        if (i == _src.size() - 1) {
                            reconstructed = reconstructed
                                + "]:(" +
                                (
                                    (_src.get(i) instanceof FConstant)
                                        ? _src.get(i).toString().split("\\.")[0]
                                        : _src.get(i).toString()
                                )
                                + ")";
                        } else {
                            reconstructed = reconstructed +
                                (
                                    (_src.get(i) instanceof FConstant)
                                        ? _src.get(i).toString().split("\\.")[0]
                                        : _src.get(i).toString()
                                );
                        }
                    } else {
                        reconstructed = reconstructed + _src.get(i).toString();
                    }
                } else {
                    reconstructed = reconstructed + "(null)";
                }
                if (i < _src.size() - ((TYPES.REGISTER[_id] == ",") ? 2 : 1)) {
                    reconstructed = reconstructed
                            + ((TYPES.REGISTER[_id].equals(">"))?"-":"")
                            + TYPES.REGISTER[_id]
                            + ((TYPES.REGISTER[_id].equals("<"))?"-":"");
                }
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public boolean dependsOn(int index){
        for(Function f : _src){
            if(f.dependsOn(index)) return true;
        }
        return false;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public abstract Tsr activate(Tsr[] inputs, int j);

    @Override
    public abstract Tsr activate(Tsr[] inputs);

    @Override
    public abstract Tsr derive(Tsr[] inputs, int index, int j);

    @Override
    public abstract Tsr derive(Tsr[] inputs, int index);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public abstract double activate(final double[] inputs, int j);

    @Override
    public abstract double activate(final double[] inputs);

    @Override
    public abstract double derive(final double[] inputs, final int index, final int j);

    @Override
    public abstract double derive(final double[] inputs, final int index);

    //==================================================================================================================

    /**
     * Responsible for handling functions with multiple inputs!
     *
     * @param inputs
     * @param j
     * @param d
     * @return
     */
    protected Tsr _tensor_activation(Tsr[] inputs, int j, int d)
    {
        /**  The code below deals with deep functions (non flat):  * */
        if (!_isFlat)
        {
            if(d>=0){
                return _apply(d, ()-> _execution(inputs, d, j, Tsr.CPU));
            } else {
                if (TYPES.isFunction(_id)) {
                    //return _apply(d, ()-> _execution(inputs, d, j, Tsr.CPU));
                    //System.out.println(_src.size());
                    return (FunctionBuilder.build(TYPES.REGISTER[_id] + "(I[" + ((j < 0) ? 0 : j) + "])", true).activate(inputs));
                } else if (TYPES.isOperation(_id)) {/*  '+', '-', 'x', '*', '%', '«', '»', ',', ...  */
                    return _apply(d, ()-> _execution(inputs, d, j, Tsr.CPU));
                }
                return _apply(d, ()-> _execution(inputs, d, j, Tsr.CPU));
            }
            /** only flat functions can be executed **/
        } else {
            /**  The following code is reached in flat functions only:  * */
            Tsr output = _execute(inputs, j, d);
            /**  Autograd-Graph will be generated below for the new GraphNode: **/
            if (d < 0 && _doAD) {
                new GraphNode(output,this, inputs, ((GraphNode)inputs[0].find(GraphNode.class)).lock());
            }
            return output;
        }
    }

    private Tsr _apply(int d, Supplier<Tsr> actor){
        Tsr out = null;
        if(d>=0){
            for (int i = 0; i < _src.size(); i++) {//constants need to be figured out!
                int di = (_src.get(i).dependsOn(d))?i:-1;
                if(di>=0){
                    if(out==null){
                        out = actor.get();
                    } else {
                        Tsr.CPU.execute(new Tsr[]{null, actor.get(), out}, TYPES.LOOKUP.get("+"), -1);
                    }
                }
            }
        } else {
            out = actor.get();
        }
        return out;
    }

    private Tsr _execute(Tsr[] inputs, int j, int d)
    {
        Device device = (Device) inputs[0].find(Device.class);
        boolean onSameDevice = _shareGuestDevice(inputs);
        boolean doAccel = (!TYPES.REGISTER[_id].equals(",") && onSameDevice);
        Device myDevice = (doAccel&&device!=null)?device:Tsr.CPU;

        if (TYPES.REGISTER[_id].equals("x")) {
            Tsr tensor1 = _src.get(0).activate(inputs).setIsVirtual(false);
            Tsr tensor2 = _src.get(1).activate(inputs).setIsVirtual(false);
            Tsr newTensor = new Tsr(Tsr.fcn.indexing.shpOfCon(tensor1.shape(), tensor2.shape()));
            Tsr[] array = new Tsr[]{newTensor, tensor1, tensor2};
            myDevice.execute(array, _id, d);
            return array[0];

        } else if (_id == TYPES.LOOKUP.get("<<x") || _id == TYPES.LOOKUP.get("x>>")) {
            if (d < 0) {
                Tsr[] tsrs = new Tsr[]{
                        _src.get(0).activate(inputs).setIsVirtual(false),
                        _src.get(1).activate(inputs).setIsVirtual(false),
                        _src.get(2).activate(inputs).setIsVirtual(false)
                };
                myDevice.execute(tsrs, _id, 0);
                if (_id == TYPES.LOOKUP.get("x>>")) return tsrs[2]; else return tsrs[0];
            }
            return null;
        } else if (TYPES.REGISTER[_id].equals(",")) {
            int[] newForm = new int[_src.size() - 1];
            for (int i = 0; i < _src.size() - 1; i++) {
                newForm[i] = (int) Tsr.fcn.io.getFrom(_src.get(i).activate(inputs), 0);
            }
            if (d >= 0) {//reverse reshape:
                int[] reversed = new int[newForm.length];
                for (int i = 0; i < newForm.length; i++) {
                    if(newForm[i]>=0){
                        reversed[newForm[i]] = i;
                    } else {//Exception! (not auto-differentiable)
                        throw new IllegalStateException("[AbstractFunction][_execute]: reshape operation cannot be reversed!");
                    }
                }
            }
            Tsr t = inputs[inputs.length - 1];
            return Tsr.fcn.exec.reshaped(t, newForm, true);//t.reshape(newForm);
        } else {
            return  _apply(d, ()-> _execution(inputs, d, j, myDevice));
        }
    }

    private Tsr _execution(Tsr[] inputs, int d, int j, Device device) {
        if (TYPES.isOperation(_id)) {/*  '+', '-', 'x', '*', '%', '«', '»', ',', ...  */
            //return _apply(d, ()-> _execution(inputs, d, j, Tsr.CPU));
            if (j < 0 && d<0 &&!_isFlat) {
                String operation = "";
                Tsr[] tsrs = _src_acti(inputs, j, d, 0);
                for (int i = 0; i < tsrs.length; i++) {
                    operation += "I["+i+"]"+((i==tsrs.length-1)?"":TYPES.REGISTER[_id]);
                }
                return (FunctionBuilder.build(operation, _doAD).activate(tsrs));
            }
        }
        Tsr[] tsrs;
        if(TYPES.isIndexer(_id)) tsrs = new Tsr[1 +inputs.length]; else tsrs = new Tsr[1 + _src.size()];
        if(d>=0) {
            //Chain-rule (forward AD):
            //inner times out means:
            //first derive source!
            //like so:
            if(TYPES.isIndexer(_id)){
                for (int i = 1; i < tsrs.length; i++) tsrs[i] = _src.get(0).derive(inputs, d, i-1);
            } else {
                for (int i = 1; i < tsrs.length; i++) tsrs[i] = (j>=0)?_src.get(i-1).derive(inputs, d, j):_src.get(i-1).derive(inputs, d);
            }
            //then add them all together! (is possible because of linearity...)
            Tsr inner;
            if(tsrs.length>2){
                Tsr.CPU.execute(tsrs, TYPES.LOOKUP.get("+"), -1);
                inner = tsrs[0];//this is now the inner derivative!
            } else {
                inner = tsrs[1];
            }
            tsrs[0] = null;
            //then activate the source like so:
            if(TYPES.isIndexer(_id)){
                for (int i = 1; i < tsrs.length; i++) tsrs[i] = _src.get(0).activate(inputs, i-1);
            } else {
                for (int i = 1; i < tsrs.length; i++) tsrs[i] = (j>=0)?_src.get(i-1).activate(inputs, j):_src.get(i-1).activate(inputs);
            }
            //get derivative index within src list:
            for(int i=0; i<_src.size(); i++){
                if(_src.get(i).dependsOn(d)&&!TYPES.isIndexer(_id)) {
                    d = i;
                    break;
                }
            }
            //Use those tensors for the outer derivative:
            device.execute(tsrs, _id, d); //(d>=0)
            //At the end:
            //multiply inner times outer:
            tsrs = new Tsr[]{null, inner, tsrs[0]};
            device.execute(tsrs, TYPES.LOOKUP.get("*"), -1);
            return tsrs[0];
        } else {
            if(TYPES.isIndexer(_id)){
                tsrs = new Tsr[1 +inputs.length];
                if(d<0){
                    for (int i = 1; i < tsrs.length; i++) tsrs[i] = _src.get(0).activate(inputs, i-1);
                } else {
                    for (int i = 1; i < tsrs.length; i++) tsrs[i] = _src.get(0).derive(inputs, d, i-1);
                }
                device.execute(tsrs, _id, d);
            } else {
                tsrs = _src_acti(inputs, j, d, 1);//new Tsr[1 + _src.size()];
                device.execute(tsrs, _id, d);
            }
        }
        return (tsrs[0]==null)?tsrs[1]:tsrs[0];

    }

    private Tsr[] _src_acti(Tsr[] inputs, int j, int d, int offset){
        int[] tempShape = null;
        Tsr[] tsrs = new Tsr[_src.size()+offset];
        for (int i = offset; i < tsrs.length; i++) {//constants need to be figured out!
            if (!(_src.get(i-offset) instanceof FConstant)) {
                if(d<0){
                    tsrs[i] = (j>=0)?_src.get(i-offset).activate(inputs, j):_src.get(i-offset).activate(inputs);
                } else {
                    tsrs[i] = (j>=0)?_src.get(i-offset).derive(inputs, d, j):_src.get(i-offset).derive(inputs, d);
                }
                tempShape = (tempShape == null) ? tsrs[i].shape() : tempShape;
            }
        }
        for (int i = offset; i < tsrs.length; i++) {
            if(tsrs[i]==null) {
                tsrs[i] =
                        (j < 0)
                                ? new Tsr(tempShape, ((FConstant) _src.get(i-offset)).value())
                                : new Tsr(tempShape,_src.get(i-offset).activate(new double[]{}, j));
            }
        }
        return tsrs;
    }

    private Tsr[] _source_activation(Tsr[] input, int j) {
        int[] tempShape = null;
        Tsr[] tsrs = new Tsr[_src.size()];
        for (int i = 0; i < tsrs.length; i++) {//constants need to be figured out!
            if (!(_src.get(i) instanceof FConstant)) {
                tsrs[i] = (j < 0) ? _src.get(i).activate(input) : _src.get(i).activate(input, j);
                tempShape = (tempShape == null) ? tsrs[i].shape() : tempShape;
            }
        }
        for (int i = 0; i < tsrs.length; i++) {
            if(tsrs[i]==null) {
                tsrs[i] =
                    (j < 0)
                        ? new Tsr(tempShape, ((FConstant) _src.get(i)).value())
                        : new Tsr(tempShape,_src.get(i).activate(new double[]{}, j));
            }
        }
        return tsrs;
    }

    private static boolean _shareGuestDevice(Tsr[] tsrs) {
        boolean onSameGuestDevice = true;
        Device device = null;
        for (int ti = 0; ti < tsrs.length; ti++) {
            device = (tsrs[ti].isOutsourced()) ? (Device) tsrs[ti].find(Device.class) : device;
        }
        if (device != null) {
            for (int ti = 0; ti < tsrs.length; ti++) {
                onSameGuestDevice = (!tsrs[ti].isVirtual() && device == tsrs[ti].find(Device.class)) && onSameGuestDevice;
            }
        } else {
            onSameGuestDevice = false;
        }
        if(device!=null && tsrs.length==2 && tsrs[1].size()==1){
            onSameGuestDevice = true;
        }
        return onSameGuestDevice;
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    protected double _scalar_activation(double input, boolean derive) {
        switch (TYPES.REGISTER[_id]) {
            case "relu":
                return exec.reLu(input, derive);
            case "sig":
                return exec.sigmoid(input, derive);
            case "tanh":
                return exec.tanh(input, derive);
            case "quad":
                return exec.quadratic(input, derive);
            case "lig":
                return exec.ligmoid(input, derive);
            case "lin":
                return exec.linear(input, derive);
            case "gaus":
                return exec.gaussian(input, derive);
            case "abs":
                return exec.absolute(input, derive);
            case "sin":
                return exec.sinus(input, derive);
            case "cos":
                return exec.cosinus(input, derive);
            default:
                return input;
        }
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    protected double _scalar_activation(double[] input, int j, int d) {
        switch (TYPES.REGISTER[_id]) {
            case "sum":
                return (j < 0) ? exec.summation(input, d, _src) : exec.summation(input, j, d, _src);
            case "prod":
                return (j < 0) ? exec.PI(input, d, _src) : exec.PI(input, j, d, _src);
            case "^":
                return (j < 0) ? exec.power(input, d, _src) : exec.power(input, j, d, _src);
            case "/":
                return (j < 0) ? exec.division(input, d, _src) : exec.division(input, j, d, _src);
            case "*":
                return (j < 0) ? exec.multiplication(input, d, _src) : exec.multiplication(input, j, d, _src);
            case "%":
                return (j < 0) ? exec.modulo(input, d, _src) : exec.modulo(input, j, d, _src);
            case "-":
                return (j < 0) ? exec.subtraction(input, d, _src) : exec.subtraction(input, j, d, _src);
            case "+":
                return (j < 0) ? exec.addition(input, d, _src) : exec.addition(input, j, d, _src);
            case "x"://convolve
                return (j < 0) ? exec.multiplication(input, d, _src) : exec.multiplication(input, j, d, _src);
            default:
                return _scalar_activation(input[0], d>=0);
        }
    }

    public static class exec
    {
        private interface Actor
        {
            void apply(Integer i, double[] v1, double[] v2);
        }

        @Contract(pure = true)
        public static void foreach(Tsr t1, Tsr t2, Actor action) {
            double[] inputValue = (t1.value64() == null) ? new double[t1.size()] : t1.value64();
            double[] outputValue = (t2.value64() == null) ? new double[t2.size()] : t2.value64();
            t1.foreach((i) -> action.apply(i, inputValue, outputValue));
            t2.setValue(outputValue);
            t1.setValue(inputValue);
        }
        //--------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        public static double reLu(double input, boolean derive) {
            double output;
            if (!derive) {
                if (input >= 0) {
                    output = (input);
                } else {
                    output = (input) * 0.01;
                }
                return output;
            } else {
                if (input >= 0) {
                    output = 1;
                } else {
                    output = 0.01;
                }
                return output;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        public static double sigmoid(double input, boolean derive) {
            if (!derive) {
                return 1 / (1 + Math.pow(Math.E, (-input)));
            } else {
                return (Math.pow(Math.E, -(input))) / (Math.pow((1 + Math.pow(Math.E, -(input))), 2) + 2 * Math.pow(Math.E, -(input)));
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        public static double tanh(double input, boolean derive) {
            if (!derive) {
                return ((input)) / Math.pow((1 + Math.pow(((input)), 2)), 0.5);
            } else {
                return (1 - Math.pow(((input) / Math.pow((1 + Math.pow((input), 2)), 0.5)), 2));
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        public static double quadratic(double input, boolean derive) {
            if (!derive) {
                return ((input) * (input));
            } else {
                return 2 * input;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        public static double ligmoid(double input, boolean derive) {
            if (!derive) {
                return (Math.log(1+Math.pow(Math.E, input)));
            } else {
                return sigmoid(input, false);
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        public static double linear(double input, boolean derive) {
            if (!derive) {
                return (input);
            } else {
                return 1;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        public static double gaussian(double input, boolean derive) {
            if (!derive) {
                return Math.pow(Math.E, -Math.pow((input), 2));
            } else {
                return -2 * ((input)) * Math.pow(Math.E, -Math.pow((input), 2));
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        public static double absolute(double input, boolean derive) {
            if (!derive) {
                return Math.abs(input);
            } else {
                return (input < 0) ? -1 : 1;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        public static double sinus(double input, boolean derive) {
            if (!derive) {
                return Math.sin(input);
            } else {
                return Math.cos(input);
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        public static double cosinus(double input, boolean derive) {
            if (!derive) {
                return Math.cos(input);
            } else {
                return -Math.sin(input);
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        private static double summation(double[] inputs, int j, int d, ArrayList<Function> src) {
            if (d < 0) {
                double sum = 0;
                boolean nothingDone = true;
                for (int i = 0; i < inputs.length; i++) {
                    sum += src.get(0).activate(inputs, i);
                    nothingDone = false;
                }
                if (nothingDone) {
                    return src.get(0).activate(inputs);
                }
                return sum;
            } else {
                return src.get(0).derive(inputs, d, j);
            }
        }

        private static double summation(double[] inputs, int d, ArrayList<Function> src) {
            if (d < 0) {
                double sum = 0;
                boolean nothingDone = true;
                for (int i = 0; i < inputs.length; i++) {
                    sum += src.get(0).activate(inputs, i);
                    nothingDone = false;
                }
                if (nothingDone) {
                    return src.get(0).activate(inputs);
                }
                return sum;
            } else {
                double sum = 0;
                boolean nothingDone = true;
                for (int i = 0; i < inputs.length; i++) {
                    sum += src.get(0).derive(inputs, d, i);
                    nothingDone = false;
                }
                if (nothingDone) {
                    return src.get(0).activate(inputs);
                }
                return sum;
            }

        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        private static double PI(double[] inputs, int j, int d, ArrayList<Function> src) {
            if (d < 0) {
                double prod = 1;
                boolean nothingDone = true;
                for (int Ii = 0; Ii < inputs.length; Ii++) {
                    prod *= src.get(0).activate(inputs, Ii);
                    nothingDone = false;
                }
                if (nothingDone) {
                    return src.get(0).activate(inputs, j);
                }
                return prod;
            } else {
                double u, ud, v, vd;
                u = src.get(0).activate(inputs, 0);
                ud = src.get(0).derive(inputs, d, 0);
                for (int ji = 1; ji < inputs.length; ji++) {
                    v = src.get(0).activate(inputs, ji);
                    vd = src.get(0).derive(inputs, d, ji);
                    ud = u * vd + v * ud;
                    u *= v;
                }
                return ud;
            }
        }

        @Contract(pure = true)
        private static double PI(double[] inputs, int d, ArrayList<Function> src) {
            if (d < 0) {
                double prod = 1;
                boolean nothingDone = true;
                for (int i = 0; i < inputs.length; i++) {
                    //if (sources.get(0).dependsOn(Ii)) {
                    prod *= src.get(0).activate(inputs, i);
                    nothingDone = false;
                    //}
                }
                if (nothingDone) {
                    return src.get(0).activate(inputs);
                }
                return prod;
            } else {
                double u, ud, v, vd;
                u = src.get(0).activate(inputs, 0);
                ud = src.get(0).derive(inputs, d, 0);
                for (int j = 1; j < inputs.length; j++) {
                    v = src.get(0).activate(inputs, j);
                    vd = src.get(0).derive(inputs, d, j);
                    ud = u * vd + v * ud;
                    u *= v;
                }
                return ud;
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        // d/dx(f(x)^g(x))=
        // f(x)^g(x) * d/dx(g(x)) * ln(f(x))
        // + f(x)^(g(x)-1) * g(x) * d/dx(f(x))
        @Contract(pure = true)
        private static double power(double[] inputs, int j, int d, ArrayList<Function> src) {
            if (d < 0) {
                double result = src.get(0).activate(inputs, j);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).activate(inputs, j);
                    result = Math.pow(result, current);
                }
                return result;
            } else {
                double out = 0;
                for(int si=0; si<src.size(); si++){
                    double b = 1;
                    for (int i = 1; i < src.size(); i++) {
                        b *= src.get(i).activate(inputs, j);
                    }
                    if(si==0){
                        out += src.get(0).derive(inputs, d, j)*b*Math.pow(src.get(0).activate(inputs, j), b-1);
                    } else {
                        double a = src.get(0).activate(inputs, j);
                        out += (a>=0)?src.get(si).derive(inputs, d, j)*b*Math.log(a):0;
                    }
                }
                return out;
            }
        }

        @Contract(pure = true)
        private static double power(double[] inputs, int d, ArrayList<Function> src) {
            if (d < 0) {
                double result = src.get(0).activate(inputs);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).activate(inputs);
                    result = Math.pow(result, current);
                }
                return result;
            } else {
                double out = 0;
                double b = 1;
                for (int i = 1; i < src.size(); i++) {
                    b *= (i==d)?1:src.get(i).activate(inputs);
                }
                for(int si=0; si<src.size(); si++){
                    if(si==0){
                        out += src.get(0).derive(inputs, d)*b*Math.pow(src.get(0).activate(inputs), b-1);
                    } else {
                        double a = src.get(0).activate(inputs);
                        out += (a>=0)?src.get(si).derive(inputs, d)*b*Math.log(a):0;
                    }
                }
                return out;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        private static double division(double[] inputs, int j, int d, ArrayList<Function> src) {
            if (d < 0) {
                double result = src.get(0).activate(inputs, j);
                for (int Vi = 1; Vi < src.size(); Vi++) {
                    final double current = src.get(Vi).activate(inputs, j);
                    result /= current;
                }
                return result;
            } else {
                double u, ud, v, vd;
                u = src.get(0).activate(inputs, j);
                ud = src.get(0).derive(inputs, d, j);
                for (int i = 0; i < src.size() - 1; i++) {
                    v = src.get(i + 1).activate(inputs, j);
                    vd = src.get(i + 1).derive(inputs, d, j);
                    ud = (ud * v - u * vd) / Math.pow(v, 2);
                    u /= v;
                }
                return ud;
            }
        }

        @Contract(pure = true)
        private static double division(double[] inputs, int d, ArrayList<Function> src) {
            if (d < 0) {
                double result = src.get(0).activate(inputs);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).activate(inputs);
                    result /= current;
                }
                return result;
            } else {
                double derivative = 0;
                double tempVar = src.get(0).activate(inputs);
                derivative = src.get(0).derive(inputs, d);

                for (int i = 0; i < src.size() - 1; i++) {
                    double u, ud, v, vd;
                    v = src.get(i + 1).activate(inputs);
                    vd = src.get(i + 1).derive(inputs, d);
                    u = tempVar;
                    ud = derivative;
                    derivative = (ud * v - u * vd) / Math.pow(v, 2);
                    tempVar /= v;
                }
                return derivative;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        private static double multiplication(double[] inputs, int j, int d, ArrayList<Function> src) {
            if (d < 0) {
                double result = src.get(0).activate(inputs, j);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).activate(inputs, j);
                    result *= current;
                }
                return result;
            } else {
                double u, ud, v, vd;
                u = src.get(0).activate(inputs, j);
                ud = src.get(0).derive(inputs, d, j);

                for (int ji = 1; ji < src.size(); ji++) {
                    v = src.get(ji).activate(inputs, j);
                    vd = src.get(ji).derive(inputs, d, j);
                    ud = u * vd + v * ud;
                    u *= v;
                }
                return ud;
            }
        }

        @Contract(pure = true)
        private static double multiplication(double[] inputs, int d, ArrayList<Function> src) {
            if (d < 0) {
                double result = src.get(0).activate(inputs);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).activate(inputs);
                    result *= current;
                }
                return result;
            } else {
                double u, ud, v, vd;
                u = src.get(0).activate(inputs);
                ud = src.get(0).derive(inputs, d);
                for (int j = 1; j < src.size(); j++) {
                    v = src.get(j).activate(inputs);
                    vd = src.get(j).derive(inputs, d);

                    ud = u * vd + v * ud;
                    u *= v;//this step can be avoided (TODO optimize)
                }
                return ud;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


        @Contract(pure = true)
        private static double idy(double[] inputs, int d, ArrayList<Function> src) {
            if (d < 0) {
                inputs[0] = inputs[1];
            } else {
            }
            return 0;
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        private static double modulo(double[] inputs, int j, int d, ArrayList<Function> src) {
            if (d < 0) {
                double result = src.get(0).activate(inputs, j);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).activate(inputs, j);
                    result %= current;
                }
                return result;
            } else {
                return src.get(0).derive(inputs, d, j);// j ?
            }
        }

        @Contract(pure = true)
        private static double modulo(double[] inputs, int d, ArrayList<Function> src) {
            if (d < 0) {
                double result = src.get(0).activate(inputs);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).activate(inputs);
                    result %= current;
                }
                return result;
            } else {
                return src.get(0).derive(inputs, d);
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        private static double subtraction(double[] inputs, int j, int d, ArrayList<Function> src) {
            if (d < 0) {
                double result = src.get(0).activate(inputs, j);
                for (int Vi = 1; Vi < src.size(); Vi++) {
                    final double current = src.get(Vi).activate(inputs, j);
                    result -= current;
                }
                return result;
            } else {
                double derivative = 0;
                for (int i = 0; i < src.size(); ++i) {
                    if (i == 0) {
                        derivative += src.get(i).derive(inputs, d, j);
                    } else {
                        derivative -= src.get(i).derive(inputs, d, j);
                    }
                }
                return derivative;
            }
        }

        @Contract(pure = true)
        private static double subtraction(double[] inputs, int d, ArrayList<Function> src) {
            if (d < 0) {
                double result = src.get(0).activate(inputs);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).activate(inputs);
                    result -= current;
                }
                return result;
            } else {
                double derivative = 0;
                for (int i = 0; i < src.size(); ++i) {
                    if (i == 0) {
                        derivative += src.get(i).derive(inputs, d);
                    } else {
                        derivative -= src.get(i).derive(inputs, d);
                    }
                }
                return derivative;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        private static double addition(double[] inputs, int j, int d, ArrayList<Function> src) {
            if (d < 0) {
                double result = src.get(0).activate(inputs, j);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).activate(inputs, j);
                    result += current;
                }
                return result;
            } else {
                double derivative = 0;
                for (int i = 0; i < src.size(); ++i) {
                    derivative += src.get(i).derive(inputs, d, j);
                }
                return derivative;
            }
        }

        @Contract(pure = true)
        private static double addition(double[] inputs, int d, ArrayList<Function> src) {
            if (d < 0) {
                double result = src.get(0).activate(inputs);
                for (int Vi = 1; Vi < src.size(); Vi++) {
                    final double current = src.get(Vi).activate(inputs);
                    result += current;
                }
                return result;
            } else {
                double derivative = 0;
                for (int i = 0; i < src.size(); ++i) {
                    derivative += src.get(i).derive(inputs, d);
                }
                return derivative;
            }
        }


    }

}
