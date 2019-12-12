package neureka.function.factory;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.function.Function;
import neureka.function.factory.assembly.FunctionBuilder;
import neureka.function.factory.autograd.GraphLock;
import neureka.function.factory.autograd.GraphNode;
import neureka.function.factory.implementations.FConstant;
import org.jetbrains.annotations.Contract;

import java.util.ArrayList;

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
            reconstructed = ((TYPES.REGISTER[_id] == ",") ? "[" : "") + reconstructed;
            for (int i = 0; i < _src.size(); ++i) {
                if (_src.get(i) != null) {
                    if ((TYPES.REGISTER[_id] == ",")) {
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
                            + ((TYPES.REGISTER[_id]==">")?"-":"")
                            + TYPES.REGISTER[_id]
                            + ((TYPES.REGISTER[_id]=="<")?"-":"");
                }
            }
        }
        return "(" + reconstructed + ")";
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
     * Responsible for handling functions with id's 0-9  (single input functions!)
     */
    protected Tsr _tensor_activation(Tsr input, boolean derive)
    {
        Tsr output = new Tsr(input, false);//Tsr.fcn.create.newTsr(input.shape(), input.translation());
        if (!derive && !_isFlat) {
            output.inject(FunctionBuilder.build(_id, 1, true).activate(new Tsr[]{input}));
            output.add(input.find(GraphLock.class));
            return output;
        }
        if (input.isOutsourced()) {
            Device device = (Device) input.find(Device.class);
            device.add(output);
            device.execute(new Tsr[]{output, input}, _id, (derive) ? 0 : -1);
        } else {
            exec.foreach(
                input, output,
                (i, inputValue, outputValue) -> outputValue[i] = _scalar_activation(inputValue[i], derive)
            );
        }
        if (!derive && _doAD) {
            new GraphNode(output,this, new Tsr[]{input}, ((GraphNode)input.find(GraphNode.class)).lock());
        }
        output.add(input.find(GraphLock.class));
        return output;
    }

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
        if (d < 0 && !_isFlat)
        {
            /** only flat functions can be executed **/
            if (TYPES.isFunction(_id)) {
                return (FunctionBuilder.build(TYPES.REGISTER[_id] + "(I[" + ((j < 0) ? 0 : j) + "])", true).activate(inputs));
            } else {
                if (TYPES.isFunction(_id)||TYPES.isIndexer(_id)) {
                    /**  SUMMATION, PI,  **/
                    Tsr[] tsrs = _source_activation(inputs);
                    return (FunctionBuilder.build(TYPES.REGISTER[_id] + "(I[j])", true).activate(tsrs));
                } else if (TYPES.isOperation(_id)) {
                    /**  '+', '-', 'x', '*', '%', '«', '»', ',', ...  **/
                    String operation = (TYPES.REGISTER[_id].length() > 1) ? TYPES.REGISTER[_id] : "";
                    Tsr[] tsrs = _source_activation(inputs, j, null);
                    for (int i = 0; i < tsrs.length; i++) {
                        operation += "I[" + i + "]" + ((i + 1 < tsrs.length) ? TYPES.REGISTER[_id] : "");
                    }
                    if (j < 0) {
                        return (FunctionBuilder.build(operation, _doAD).activate(tsrs));
                    } else {
                        return (FunctionBuilder.build(operation, _doAD).activate(tsrs, j));
                    }
                } else {
                    /**  Tensor shape translation:  **/
                    Tsr[] tsrs = _source_activation(inputs, j, new int[]{1});
                    if (j < 0) {
                        return FunctionBuilder.build(_id, tsrs.length, _doAD).activate(tsrs);
                    } else {
                        return FunctionBuilder.build(_id, tsrs.length, _doAD).activate(tsrs, j);
                    }
                }
            }
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

    private Tsr _execute(Tsr[] inputs, int j, int d)
    {
        Device device = (Device) inputs[0].find(Device.class);
        boolean onSameDevice = _shareGuestDevice(inputs) && TYPES.REGISTER[_id] != "," && !(TYPES.isConvection(_id) && d > -1);
        if (onSameDevice)
        {
            Tsr[] tsrs = new Tsr[1 + _src.size()];
            int new_d = d;
            boolean adjusted = false;
            for (int i = 1; i < tsrs.length; i++) {
                tsrs[i] = _src.get(i-1).activate(inputs);
                if(!adjusted && d>=0 && inputs[d]==tsrs[i]){
                    new_d = i-1;//The index of the derivative is adjusted here.
                    adjusted = true;// ...this occurs when source nodes are constants!
                }
            }
            device.execute(tsrs, _id, new_d);
            return (tsrs[0]==null)?tsrs[1]:tsrs[0];
        } else {
            if (TYPES.REGISTER[_id] == "x") {
                if (d < 0) {
                    Tsr tensor1 = _src.get(0).activate(inputs).setIsVirtual(false);
                    Tsr tensor2 = _src.get(1).activate(inputs).setIsVirtual(false);
                    Tsr newTensor = new Tsr(Tsr.fcn.indexing.shpOfCon(tensor1.shape(), tensor2.shape()));
                    exec.convolve_multiply(newTensor, tensor1, tensor2, -1);
                    return newTensor;
                } else {
                    if (d == 0) {
                        return (_src.get(1).activate(inputs));
                    } else {
                        return (_src.get(0).activate(inputs));
                    }
                }
            } else if (_id == TYPES.LOOKUP.get("<<x") || _id == TYPES.LOOKUP.get("x>>")) {
                if (d < 0) {
                    if (_id == TYPES.LOOKUP.get("x>>")) {
                        Tsr out = _src.get(2).activate(inputs);
                        exec.convolve_multiply(
                                out.setIsVirtual(false),
                                _src.get(1).activate(inputs).setIsVirtual(false),
                                _src.get(0).activate(inputs).setIsVirtual(false),
                                0
                        );
                        return out;
                    } else {
                        Tsr out = _src.get(0).activate(inputs);
                        exec.convolve_multiply(
                                out.setIsVirtual(false),
                                _src.get(1).activate(inputs).setIsVirtual(false),
                                _src.get(2).activate(inputs).setIsVirtual(false),
                                0
                        );
                        return out;
                    }
                } else {//Todo: What then? :
                    if (d == 0) {
                        return (_src.get(1).activate(inputs));
                    } else {
                        return (_src.get(0).activate(inputs));
                    }
                }
            } else if (TYPES.REGISTER[_id] == ",") {
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
            } else if(TYPES.REGISTER[_id]=="<" || TYPES.REGISTER[_id]==">") {
                //return _src.get(0).activate(inputs).setValue64(_src.get(1).activate(inputs).value64(true));
                Tsr output = (TYPES.REGISTER[_id]=="<")?_src.get(0).activate(inputs):_src.get(1).activate(inputs);
                Tsr input =  (TYPES.REGISTER[_id]=="<")?_src.get(1).activate(inputs):_src.get(0).activate(inputs);
                output.foreach((i) -> {
                    output.value64()[Tsr.fcn.indexing.i_of_i(i, output)] = input.value64()[Tsr.fcn.indexing.i_of_i(i, input)];
                });
                return output;
            } else {
                double[] inp = new double[inputs.length];
                Tsr output = new Tsr(inputs[0], false);
                //Tsr finalOutput = output;
                double[][] data = new double[inputs.length][];
                for(int i=0; i<data.length; i++){
                    data[i] = inputs[i].value64();
                }
                if(output.is64()){
                    double[] outputValue = output.value64();//.value64();
                    output.foreach((i) -> {
                        for (int ii = 0; ii < inputs.length; ii++) {
                            inp[ii] = data[ii][Tsr.fcn.indexing.i_of_i(i, inputs[ii])];//ids[ii]];//i
                        }
                        outputValue[Tsr.fcn.indexing.i_of_i(i, output)] = _scalar_activation(inp, j, d);
                    });
                } else {
                    float[] outputValue = output.value32();//.value64();
                    output.foreach((i) -> {
                        for (int ii = 0; ii < inputs.length; ii++) {
                            inp[ii] = data[ii][Tsr.fcn.indexing.i_of_i(i, inputs[ii])];//ids[ii]];//i
                        }
                        outputValue[Tsr.fcn.indexing.i_of_i(i, output)] = (float)_scalar_activation(inp, j, d);
                    });
                }
                return  output;
            }
        }
        //Todo: warning/exception.....
        //return new Tsr(inputs[0], false);//Tsr.fcn.create.newTsr(inputs[0].shape(), inputs[0].translation());
    }

    /**
     * **/
    private Tsr[] _source_activation(Tsr[] input) {
        Tsr[] tsrs = new Tsr[input.length];
        for (int i = 0; i < tsrs.length; i++) {
            tsrs[i] = _src.get(0).activate(input, i);
        }
        return tsrs;
    }

    private Tsr[] _source_activation(Tsr[] input, int j, int[] templateShape) {
        boolean shareDevice = _shareGuestDevice(input);
        Tsr[] tsrs = new Tsr[_src.size()];
        for (int i = 0; i < tsrs.length; i++) {//constants need to be figured out!
            if (_src.get(i) instanceof FConstant) {
                tsrs[i] = null;
            } else {
                tsrs[i] = (j < 0) ? _src.get(i).activate(input) : _src.get(i).activate(input, j);
                templateShape =
                        (templateShape == null)
                                ? tsrs[i].shape()
                                : templateShape;
            }
        }
        for (int i = 0; i < tsrs.length; i++) {
            tsrs[i] =
                    (tsrs[i] != null)
                            ? tsrs[i]
                            : (j < 0)
                            ? new Tsr(templateShape, ((FConstant) _src.get(i)).value())//Tsr.fcn.create.newTsr(((FConstant) _src.get(i)).value(), templateShape)
                            : new Tsr(templateShape,_src.get(i).activate(new double[]{}, j));//Tsr.fcn.create.newTsr(_src.get(i).activate(new double[]{}, j), templateShape);
        }
        if(shareDevice){
            Device shared = (Device) tsrs[0].find(Device.class);
            if(tsrs.length>2){// Constant sources will be converted into full Tensors and stored on the gpu!
                for(int i=0; i<tsrs.length; i++){
                    if(!tsrs[i].isOutsourced()){
                        shared.add(tsrs[i]);
                    }
                }
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
        switch (_id) {
            case 0:
                return exec.reLu(input, derive);
            case 1:
                return exec.sigmoid(input, derive);
            case 2:
                return exec.tanh(input, derive);
            case 3:
                return exec.quadratic(input, derive);
            case 4:
                return exec.ligmoid(input, derive);
            case 5:
                return exec.linear(input, derive);
            case 6:
                return exec.gaussian(input, derive);
            case 7:
                return exec.absolute(input, derive);
            case 8:
                return exec.sinus(input, derive);
            case 9:
                return exec.cosinus(input, derive);
            default:
                return input;
        }
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    protected double _scalar_activation(double[] input, int j, int d) {
        switch (_id) {
            case 10:
                return (j < 0) ? exec.summation(input, d, _src) : exec.summation(input, j, d, _src);
            case 11:
                return (j < 0) ? exec.PI(input, d, _src) : exec.PI(input, j, d, _src);
            case 12:
                return (j < 0) ? exec.power(input, d, _src) : exec.power(input, j, d, _src);
            case 13:
                return (j < 0) ? exec.division(input, d, _src) : exec.division(input, j, d, _src);
            case 14:
                return (j < 0) ? exec.multiplication(input, d, _src) : exec.multiplication(input, j, d, _src);
            case 15:
                return (j < 0) ? exec.modulo(input, d, _src) : exec.modulo(input, j, d, _src);
            case 16:
                return (j < 0) ? exec.subtraction(input, d, _src) : exec.subtraction(input, j, d, _src);
            case 17:
                return (j < 0) ? exec.addition(input, d, _src) : exec.addition(input, j, d, _src);
            case 18://convolve_template
                return (j < 0) ? exec.multiplication(input, d, _src) : exec.multiplication(input, j, d, _src);
            //case 19://inv left
            //    return (j < 0) ? exec.multiplication(input, d, _src) : exec.multiplication(input, j, d, _src);
            //case 20://inv right
            //    return (j < 0) ? exec.multiplication(input, d, _src) : exec.multiplication(input, j, d, _src);
            //case 21:// reshape
            //    return (j < 0) ? exec.multiplication(input, d, _src) : exec.multiplication(input, j, d, _src);
            //case 22://idy left
            //    return (j < 0) ? exec.idy(input, d, _src) : exec.idy(input, d, _src);
            //case 23://idy right
            //    return (j < 0) ? exec.idy(new double[]{input[1], input[0]}, d, _src) : exec.idy(new double[]{input[1], input[0]}, d, _src);
            default:
                return 0;
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

        interface Range {
            void execute(int start, int end);
        }

        private static void threaded(int sze, Range range){
            boolean doThreading = false;
            if(sze>128){
                doThreading = ((sze/Runtime.getRuntime().availableProcessors()) > 32);
            }
            if(!doThreading){
                range.execute(0, sze);
            } else {
                int threadCount = Runtime.getRuntime().availableProcessors();
                final int chunk=(sze/threadCount);
                Thread[] th = new Thread[threadCount];
                for(int i=0;i<threadCount;i++){
                    final int start = i*chunk;
                    final  int end = (i==threadCount-1)?sze:((i+1)*chunk);
                    th[i]=new Thread(()->{
                        range.execute(start, end);
                    });
                    th[i].start();
                }
                for(int i=0;i<threadCount;i++){
                    try {
                        th[i].join();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }

        public static void convolve_multiply(
                Tsr t0_drain,
                Tsr t1_source,
                Tsr t2_source,
                int d
        ){
            double[] t1_val = t1_source.value64();
            double[] t2_val = t2_source.value64();
            Operator operation;
            if(d<0){
                operation = (t0Idx, t1Idx, t2Idx)->{
                    return t1_val[Tsr.fcn.indexing.i_of_idx(t1Idx, t1_source)] * t2_val[Tsr.fcn.indexing.i_of_idx(t2Idx, t2_source)];
                };
            } else {
                operation = (t0Idx, t1Idx, t2Idx)->{
                    return t1_val[Tsr.fcn.indexing.i_of_idx(t1Idx, t1_source)] * t2_val[Tsr.fcn.indexing.i_of_idx(t2Idx, t2_source)];
                };
            }
            threaded(t0_drain.size(), ((start, end) -> {
                convolve_template(
                        t0_drain, t1_source, t2_source, d,
                        start, end,
                        operation
                );
            }));
        }

        @Contract(pure = true)
        public static void convolve_template(
                Tsr t0_drain, Tsr t1_source, Tsr t2_source,
                int d,
                int i, int end,
                Operator operation
        ){
            int[] t0Shp = t0_drain.shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
            int[] t1Shp = t1_source.shape();
            int[] t2Shp = t2_source.shape();
            int rank = t0Shp.length;
            int[] t0Idx = new int[rank];
            int[] t1Idx = new int[rank];
            int[] t2Idx = new int[rank];
            double[] t0_value = t0_drain.value64();
            //double[] t1_value = t1_source.value64();
            //double[] t2_value = t2_source.value64();
            //int drnSze = t0_drain.size();
            //int i = 0;

            if(d<0){
                while (i < end)//drnSze)
                {//increment on drain accordingly:
                    int ri = 0;
                    while (ri < rank) {
                        if (t1Shp[ri] == t2Shp[ri]) {
                            t1Idx[ri] = t0Idx[ri];
                            t2Idx[ri] = t0Idx[ri];
                        } else if (t1Shp[ri] > t2Shp[ri]) {
                            t1Idx[ri] = t0Idx[ri];
                            t2Idx[ri] = 0;
                        } else if (t1Shp[ri] < t2Shp[ri]) {
                            t1Idx[ri] = 0;
                            t2Idx[ri] = t0Idx[ri];
                        }
                        ri++;
                    }
                    //----------
                    // multiplication:
                    double value = 0;
                    boolean running = true;
                    boolean incrementing = false;
                    while (running) {
                        ri = (ri == rank) ? 0 : ri;
                        if (!incrementing) {
                            value += operation.execute(t0Idx, t1Idx, t2Idx);
                                //t1_value[Tsr.fcn.indexing.i_of_idx(t1Idx, t1_source)]
                                //    * t2_value[Tsr.fcn.indexing.i_of_idx(t2Idx, t2_source)];
                            incrementing = true;
                            ri = 0;
                        } else {//incrementing:
                            if (t1Idx[ri] < t1Shp[ri] && t2Idx[ri] < t2Shp[ri]) {
                                t1Idx[ri]++;
                                t2Idx[ri]++;
                                if (t1Idx[ri] == t1Shp[ri] || t2Idx[ri] == t2Shp[ri]) {
                                    running = running && !(ri == (rank - 1));
                                    if (t1Shp[ri] == t2Shp[ri]) {
                                        t1Idx[ri] = t0Idx[ri];
                                        t2Idx[ri] = t0Idx[ri];
                                    } else if (t1Shp[ri] > t2Shp[ri]) {
                                        t1Idx[ri] = t0Idx[ri];
                                        t2Idx[ri] = 0;
                                    } else if (t1Shp[ri] < t2Shp[ri]) {
                                        t1Idx[ri] = 0;
                                        t2Idx[ri] = t0Idx[ri];
                                    }
                                    ri++;
                                } else {
                                    incrementing = false;
                                }
                            } else {
                                ri++;
                            }
                        }
                    }//setInto _value in drn:
                    t0_value[Tsr.fcn.indexing.i_of_idx(t0Idx, t0_drain)] = value;
                    //increment on drain:
                    Tsr.fcn.indexing.increment(t0Idx, t0Shp);
                    i++;
                }
            }
            else//---
            {
                while (i < end) {//increment on drain accordingly:
                    int ri = 0;
                    while (ri < rank) {
                        if (t2Idx[ri] == t2Shp[ri]) {//setting 0
                            t1Idx[ri] = t0Idx[ri];
                            t2Idx[ri] = 0;//mtch[mi];
                        } else {
                            if (t0Shp[ri] > t1Shp[ri]) {
                                t1Idx[ri] = (t0Idx[ri] - t2Idx[ri]);
                            } else {
                                t1Idx[ri] = (t0Idx[ri] + t2Idx[ri]);
                            }
                        }
                        ri++;
                    }
                    //----------
                    // multiplication:
                    double value = 0;
                    boolean running = true;
                    boolean incrementing = false;
                    while (running) {
                        ri = (ri == rank) ? 0 : ri;
                        if (!incrementing) {
                            boolean isMatch = true;
                            for (int rii = 0; rii < rank; rii++) {
                                isMatch = (t1Idx[rii] < t1Shp[rii] && t1Idx[rii] >= 0) && isMatch;
                            }
                            if (isMatch) {
                                value += operation.execute(t0Idx, t1Idx, t2Idx);
                            }
                            incrementing = true;
                            ri = 0;
                        } else {//incrementing:
                            if (t2Idx[ri] < t2Shp[ri]) {
                                t2Idx[ri]++;
                                if (t2Idx[ri] == t2Shp[ri]) {
                                    running = running && !(ri == (rank - 1));
                                    t1Idx[ri] = t0Idx[ri];
                                    t2Idx[ri] = 0;
                                    ri++;
                                } else {
                                    if (t0Shp[ri] > t1Shp[ri]) {
                                        t1Idx[ri] = (t0Idx[ri] - t2Idx[ri]);
                                    } else {
                                        t1Idx[ri] = (t0Idx[ri] + t2Idx[ri]);
                                    }
                                    incrementing = false;
                                }
                            } else {
                                ri++;
                            }
                        }
                    }
                    //set value in drn:
                    t0_value[Tsr.fcn.indexing.i_of_idx(t0Idx, t0_drain)] = value;
                    //increment on drain:
                    Tsr.fcn.indexing.increment(t0Idx, t0Shp);
                    i++;
                }
            }

        }

        interface Operator{
            double execute(int[] t0Idx, int[] t1Idx, int[] t2Idx);
        }

        public static void broadcast_multiply(
                Tsr t0_drain,
                Tsr t1_source,
                Tsr t2_source,
                int d
        ){
            double[] t1_val = t1_source.value64();
            double[] t2_val = t2_source.value64();
            Operator operation;
            if(d<0){
                operation = (t0Idx, t1Idx, t2Idx)->{
                    return t1_val[Tsr.fcn.indexing.i_of_idx(t1Idx, t1_source)] * t2_val[Tsr.fcn.indexing.i_of_idx(t2Idx, t2_source)];
                };
            } else {
                operation = (t0Idx, t1Idx, t2Idx)->{
                    return t1_val[Tsr.fcn.indexing.i_of_idx(t1Idx, t1_source)] * t2_val[Tsr.fcn.indexing.i_of_idx(t2Idx, t2_source)];
                };
            }
            threaded(t0_drain.size(), (start, end)->{
                broadcast_template(
                        t0_drain, t1_source, t2_source, d,
                        start, end,
                        operation
                );
            });
        }

        @Contract(pure = true)
        public static void broadcast_template(
                Tsr t0_drain, Tsr t1_source, Tsr t2_source,
                int d,
                int i, int end,
                Operator operation
        ){
            int[] t0Shp = t0_drain.shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
            int[] t1Shp = t1_source.shape();
            int[] t2Shp = t2_source.shape();
            int rank = t0Shp.length;
            int[] t0Idx = Tsr.fcn.indexing.idx_of_i(i, t0_drain);//new int[rank];
            int[] t1Idx = new int[rank];
            int[] t2Idx = new int[rank];
            double[] t0_value = t0_drain.value64();
            //double[] t1_value = t1_source.value64();
            //double[] t2_value = t2_source.value64();
            //int drnSze = t0_drain.size();
            //int i = 0;
            if(d<0){
                while (i < end)
                {//increment on drain accordingly:
                    int ri = 0;
                    while (ri < rank) {
                        if (t1Shp[ri] == t2Shp[ri]) {//Equal shapes -> out index is t1 & t2 index!for this ri
                            t1Idx[ri] = t0Idx[ri];
                            t2Idx[ri] = t0Idx[ri];
                        } else if (t1Shp[ri] > t2Shp[ri]) {//Current shape axis of t2 must be 1 !
                            t1Idx[ri] = t0Idx[ri];
                            t2Idx[ri] = 0;//...therefore it can be set to 0!
                        } else if (t1Shp[ri] < t2Shp[ri]) {//same principle:
                            t1Idx[ri] = 0;
                            t2Idx[ri] = t0Idx[ri];
                        }
                        ri++;
                    }
                    //----------
                    //setInto _value in drn:
                    t0_value[Tsr.fcn.indexing.i_of_idx(t0Idx, t0_drain)] =
                        operation.execute(t0Idx, t1Idx, t2Idx);

                    //increment on drain:
                    Tsr.fcn.indexing.increment(t0Idx, t0Shp);
                    i++;
                }
            }
            else//---//Note: src2 is now former drain!
            {
                while (i < end) {//increment on drain accordingly:
                    int ri = 0;
                    while (ri < rank) {
                        if(t0Shp[ri] == t1Shp[ri]){
                            t1Idx[ri] = t0Idx[ri];//all shapes are equal -> shape index can be inherited from origin!
                            t2Idx[ri] = t0Idx[ri];
                        } else if (t0Shp[ri] > t1Shp[ri]) {
                            t1Idx[ri] = 0;//Current origin index is larger: index can be inherited!
                            t2Idx[ri] = t0Idx[ri];
                        }
                        ri++;
                    }
                    //----------
                    // multiplication:
                    double value = 0;
                    boolean running = true;
                    boolean incrementing = false;
                    while (running) {
                        ri = (ri == rank) ? 0 : ri;
                        if (!incrementing) {
                            value +=  operation.execute(t0Idx, t1Idx, t2Idx);
                            incrementing = true;
                            ri = 0;
                        } else {//incrementing:
                            if (t0Shp[ri] < t1Shp[ri]) {//Only if origin shape is smaller than handle and drain!
                                t1Idx[ri]++;
                                t2Idx[ri]++;
                                if (t1Idx[ri] == t1Shp[ri]) {
                                    t1Idx[ri] = 0;
                                    t2Idx[ri] = 0;
                                    ri++;
                                    running = running && !(ri == (rank - 1));
                                } else {
                                    incrementing = false;//return to calculation!
                                }
                            } else {
                                ri++;
                            }
                        }
                    }
                    //set value in drn:
                    t0_value[Tsr.fcn.indexing.i_of_idx(t0Idx, t0_drain)] = value;
                    //increment on drain:
                    Tsr.fcn.indexing.increment(t0Idx, t0Shp);
                    i++;
                }
            }
        }

    }

}
