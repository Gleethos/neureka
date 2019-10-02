package neureka.core.function.factory;

import neureka.core.Tsr;
import neureka.core.function.IFunction;
import neureka.core.function.factory.assembly.FunctionBuilder;
import neureka.core.function.factory.autograd.GraphLock;
import neureka.core.function.factory.implementations.FConstant;
import neureka.core.function.factory.autograd.GraphBuilder;
import neureka.core.device.Device;
import org.jetbrains.annotations.Contract;

import java.util.ArrayList;

public abstract class Function implements IFunction {

    protected int _id;
    protected boolean _isFlat;
    protected boolean _doAD;
    protected ArrayList<IFunction> _src;

    /**
     * @param f_id
     * @param isFlat
     * @param source
     * @param doAD
     */
    protected Function(int f_id, boolean isFlat, ArrayList<IFunction> source, boolean doAD) {
        _id = f_id;
        _isFlat = isFlat;
        _src = source;
        _doAD = doAD;
    }

    @Override
    public IFunction newBuild(String expression) {
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
        Tsr output = Tsr.factory.newTensor(input.shape(), input.translation());
        if (!derive && !_isFlat) {
            output.inject(FunctionBuilder.build(_id, 1, true).activate(new Tsr[]{input}));
            output.add(input.find(GraphLock.class));
            return output;
        }
        if (input.isOutsourced()) {
            Device device = (Device) input.find(Device.class);
            device.add(output);
            device.calculate(new Tsr[]{output, input}, _id, (derive) ? 0 : -1);
        } else {
            exec.foreach(input, output, (i, inputValue, outputValue) -> {
                outputValue[i] = _scalar_activation(inputValue[i], derive);
            });
        }
        if (!derive && _doAD) {
            GraphBuilder.connect(output, new Tsr[]{input}, this);
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
    protected Tsr _tensor_activation(Tsr[] inputs, int j, int d) {
        /**  The code below deals with deep functions (non flat):  * */
        if (d < 0 && !_isFlat)
        {//only flat functions can be executed
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
        }
        /**  The following code is reached in flat functions only:  * */
        Tsr output = _execute(inputs, j, d);
        /**  Autograd-Graph will be generated below for the new GraphNode: **/
        if (d < 0 && _doAD) {
            GraphBuilder.connect(output, inputs, this);
        }
        return output;
    }

    private Tsr _execute(Tsr[] input, int j, int d)
    {
        Device device = (Device) input[0].find(Device.class);
        boolean onSameDevice = _shareGuestDevice(input);
        if (onSameDevice && TYPES.REGISTER[_id] != "," && !(TYPES.isConvection(_id) && d > -1)) {
            if(TYPES.REGISTER[_id]=="<") {
                device.overwrite(_src.get(0).activate(input), _src.get(1).activate(input));
            } else if(TYPES.REGISTER[_id]==">") {
                device.overwrite(_src.get(1).activate(input), _src.get(0).activate(input));
            } else {
                int[] shp = (TYPES.REGISTER[_id] == "x")
                    ? Tsr.factory.util.shpOfCon(_src.get(0).activate(input).shape(), _src.get(1).activate(input).shape())
                    :_src.get(0).activate(input).shape();
                Tsr output = new Tsr(shp, 0.0);
                if (device != null) {
                    device.add(output);
                }
                Tsr[] tsrs = new Tsr[1 + _src.size()];//input.length];
                tsrs[0] = output;
                for (int ii = 1; ii < tsrs.length; ii++) {
                    tsrs[ii] = _src.get(ii-1).activate(input);//input[ii - 1];
                }
                if (tsrs.length == 2 && (tsrs[0].isVirtual() || tsrs[1].isVirtual())) {
                    if (tsrs[0].isVirtual()) {
                        device.calculate(tsrs[1], tsrs[0].value()[0], _id);
                    } else {
                        device.calculate(tsrs[0], tsrs[1].value()[0], _id);
                    }
                } else {
                    device.calculate(tsrs, _id, d);
                }
                return output;
            }
        } else {
            if (TYPES.REGISTER[_id] == "x") {
                if (d < 0) {
                    return exec.convection(input[0], input[1]);
                } else {
                    if (d == 0) {
                        return (input[1]);
                    } else {
                        return (input[0]);
                    }
                }
            } else if (_id == TYPES.LOOKUP.get("<<") || _id == TYPES.LOOKUP.get(">>")) {
                if (d < 0) {
                    if (_id == TYPES.LOOKUP.get(">>")) {
                        return exec.convection_inv(
                                _src.get(0).activate(input),
                                _src.get(1).activate(input),
                                _src.get(2).activate(input),
                                false
                        );
                    } else {
                        return exec.convection_inv(
                                _src.get(2).activate(input),
                                _src.get(1).activate(input),
                                _src.get(0).activate(input),
                                false
                        );
                    }
                } else {//Todo: What then? :
                    if (d == 0) {
                        return input[1];
                    } else {
                        return input[0];
                    }
                }
            } else if (TYPES.REGISTER[_id] == ",") {
                int[] newForm = new int[input.length - 1];
                for (int i = 0; i < input.length - 1; i++) {
                    newForm[i] = (int) Tsr.factory.io.getFrom(input[i], 0);
                }
                if (d < 0) {
                    Tsr t = input[input.length - 1];
                    return Tsr.factory.exec.reshaped(t, newForm, true);//t.reshape(newForm);
                } else {//reverse reshape:
                    int[] reversed = new int[newForm.length];
                    for (int i = 0; i < newForm.length; i++) {
                        if(newForm[i]>=0){
                            reversed[newForm[i]] = i;
                        } else if(_doAD){
                            //TODO: exception! (not auto-differentiable)
                        }
                    }
                }
            } else if(TYPES.REGISTER[_id]=="<") {
                return _src.get(0).activate(input).setTargetValue(_src.get(1).activate(input).targetValue(true));
            } else if(TYPES.REGISTER[_id]==">") {
                return _src.get(1).activate(input).setTargetValue(_src.get(0).activate(input).targetValue(true));
            } else {
                double[] inp = new double[input.length];
                Tsr output = Tsr.factory.newTensor(input[0].shape(), input[0].translation());
                Tsr finalOutput = output;
                output.foreach((i) -> {
                    for (int ii = 0; ii < input.length; ii++) {
                        inp[ii] = input[ii].value()[i];
                    }

                    finalOutput.value()[i] = _scalar_activation(inp, j, d);
                });
                return  output;
            }
        }
        //Todo: warning/exception.....
        return Tsr.factory.newTensor(input[0].shape(), input[0].translation());
    }

    private Tsr[] _source_activation(Tsr[] input) {
        Tsr[] tsrs = new Tsr[input.length];
        for (int i = 0; i < tsrs.length; i++) {
            tsrs[i] = _src.get(0).activate(input, i);
        }
        return tsrs;
    }

    private Tsr[] _source_activation(Tsr[] input, int j, int[] templateShape) {
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
                            ? Tsr.factory.newTensor(((FConstant) _src.get(i)).value(), templateShape)
                            : Tsr.factory.newTensor(_src.get(i).activate(new double[]{}, j), templateShape);
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
        //if(device!=null && tsrs.length==2 && tsrs[1].size()==1){
        //    onSameGuestDevice = true;
        //}
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
            case 18:
                return (j < 0) ? exec.multiplication(input, d, _src) : exec.multiplication(input, j, d, _src);
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
            double[] inputValue = (t1.value() == null) ? new double[t1.size()] : t1.value();
            double[] outputValue = (t2.value() == null) ? new double[t2.size()] : t2.value();
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
        @Contract(pure = true)
        private static double summation(double[] input, int j, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double sum = 0;
                boolean nothingDone = true;
                for (int Ii = 0; Ii < input.length; Ii++) {
                    sum += Variable.get(0).activate(input, Ii);
                    nothingDone = false;
                }
                if (nothingDone) {
                    return Variable.get(0).activate(input);
                }
                return sum;
            } else {
                return Variable.get(0).derive(input, d, j);
            }
        }

        @Contract(pure = true)
        private static double summation(double[] input, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double sum = 0;
                boolean nothingDone = true;
                for (int Ii = 0; Ii < input.length; Ii++) {
                    sum += Variable.get(0).activate(input, Ii);
                    nothingDone = false;
                }
                if (nothingDone) {
                    return Variable.get(0).activate(input);
                }
                return sum;
            } else {
                return Variable.get(0).derive(input, d);
            }

        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        private static double PI(double[] input, int j, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double prod = 1;
                boolean nothingDone = true;
                for (int Ii = 0; Ii < input.length; Ii++) {
                    prod *= Variable.get(0).activate(input, Ii);
                    nothingDone = false;
                }
                if (nothingDone) {
                    return Variable.get(0).activate(input, j);
                }
                return prod;
            } else {
                double u, ud, v, vd;
                u = Variable.get(0).activate(input, 0);
                ud = Variable.get(0).derive(input, d, 0);
                System.out.println(ud);
                for (int ji = 1; ji < input.length; ji++) {
                    v = Variable.get(0).activate(input, ji);
                    vd = Variable.get(0).derive(input, d, ji);
                    ud = u * vd + v * ud;
                    u *= v;
                }
                return ud;
            }
        }

        @Contract(pure = true)
        private static double PI(double[] input, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double prod = 1;
                boolean nothingDone = true;
                for (int Ii = 0; Ii < input.length; Ii++) {
                    //if (sources.get(0).dependsOn(Ii)) {
                    prod *= Variable.get(0).activate(input, Ii);
                    nothingDone = false;
                    //}
                }
                if (nothingDone) {
                    return Variable.get(0).activate(input);
                }
                return prod;
            } else {
                double u, ud, v, vd;
                u = Variable.get(0).activate(input, 0);
                ud = Variable.get(0).derive(input, d, 0);
                System.out.println(ud);
                for (int j = 1; j < input.length; j++) {
                    v = Variable.get(0).activate(input, j);
                    vd = Variable.get(0).derive(input, d, j);
                    ud = u * vd + v * ud;
                    u *= v;
                }
                return ud;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        private static double power(double[] input, int j, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double result = Variable.get(0).activate(input, j);
                for (int Vi = 1; Vi < Variable.size(); Vi++) {
                    final double current = Variable.get(Vi).activate(input, j);
                    result = Math.pow(result, current);
                }
                return result;
            } else {
                // d/dx(f(x)^g(x))=
                //	f(x)^g(x) * d/dx(g(x)) * ln(f(x))
                //	+ f(x)^(g(x)-1) * g(x) * d/dx(f(x))
                double fg, dg, lnf;
                double f = Variable.get(0).activate(input, j);
                double df = Variable.get(0).derive(input, d, j);
                double g;
                for (int i = 0; i < Variable.size() - 2; i++) {
                    g = Variable.get(i + 1).activate(input, j);
                    fg = f * g;
                    dg = Variable.get(i + 1).derive(input, d, j);
                    lnf = Math.log(f);
                    df = fg * dg * lnf + f * (g - 1) * g * df;
                    f = Math.pow(f, g);
                }
                return df;
            }
        }

        @Contract(pure = true)
        private static double power(double[] input, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double result = Variable.get(0).activate(input);
                for (int Vi = 1; Vi < Variable.size(); Vi++) {
                    final double current = Variable.get(Vi).activate(input);
                    result = Math.pow(result, current);
                }
                return result;
            } else {
                // d/dx(f(x)^g(x))=
                //f(x)^g(x) * d/dx( g(x) ) * ln( f(x) )
                //+ f(x)^( g(x)-1 ) * g(x) * d/dx( f(x) )
                double fg, dg, lnf;
                double f = Variable.get(0).activate(input);
                double df = Variable.get(0).derive(input, d);
                double g = Variable.get(1).activate(input);
                df = f * g;
                for (int i = 0; i < Variable.size() - 2; i++) {
                    g = Variable.get(i + 1).activate(input);
                    fg = f * g;
                    dg = Variable.get(i + 1).derive(input, d);
                    lnf = Math.log(f);
                    df = fg * dg * lnf + f * (g - 1) * g * df;
                    f = Math.pow(f, g);
                }
                return df;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        private static double division(double[] input, int j, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double result = Variable.get(0).activate(input, j);
                for (int Vi = 1; Vi < Variable.size(); Vi++) {
                    final double current = Variable.get(Vi).activate(input, j);
                    result /= current;
                }
                return result;
            } else {
                double u, ud, v, vd;
                u = Variable.get(0).activate(input, j);
                ud = Variable.get(0).derive(input, d, j);
                for (int i = 0; i < Variable.size() - 1; i++) {
                    v = Variable.get(i + 1).activate(input, j);
                    vd = Variable.get(i + 1).derive(input, d, j);
                    ud = (ud * v - u * vd) / Math.pow(v, 2);
                    u /= v;
                }
                return ud;
            }
        }

        @Contract(pure = true)
        private static double division(double[] input, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double result = Variable.get(0).activate(input);
                for (int Vi = 1; Vi < Variable.size(); Vi++) {
                    final double current = Variable.get(Vi).activate(input);
                    result /= current;
                }
                return result;
            } else {
                double derivative = 0;
                double tempVar = Variable.get(0).activate(input);
                derivative = Variable.get(0).derive(input, d);

                for (int i = 0; i < Variable.size() - 1; i++) {
                    double u, ud, v, vd;
                    v = Variable.get(i + 1).activate(input);
                    vd = Variable.get(i + 1).derive(input, d);
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
        private static double multiplication(double[] input, int j, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double result = Variable.get(0).activate(input, j);
                for (int Vi = 1; Vi < Variable.size(); Vi++) {
                    final double current = Variable.get(Vi).activate(input, j);
                    result *= current;
                }
                return result;
            } else {
                double u, ud, v, vd;
                u = Variable.get(0).activate(input, j);
                ud = Variable.get(0).derive(input, d, j);

                for (int ji = 1; ji < Variable.size(); ji++) {
                    v = Variable.get(ji).activate(input, j);
                    vd = Variable.get(ji).derive(input, d, j);
                    //System.out.println("ud" + (u * vd + v * ud) + "=u" + u + "*vd" + vd + "+v" + v + "*ud" + ud);
                    ud = u * vd + v * ud;
                    u *= v;
                }
                //System.out.println("* d: " + ud + "; j: " + j);
                return ud;
            }
        }

        @Contract(pure = true)
        private static double multiplication(double[] input, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double result = Variable.get(0).activate(input);
                for (int Vi = 1; Vi < Variable.size(); Vi++) {
                    final double current = Variable.get(Vi).activate(input);
                    result *= current;
                }
                return result;
            } else {
                double u, ud, v, vd;
                u = Variable.get(0).activate(input);
                ud = Variable.get(0).derive(input, d);

                for (int j = 1; j < Variable.size(); j++) {
                    v = Variable.get(j).activate(input);
                    vd = Variable.get(j).derive(input, d);

                    ud = u * vd + v * ud;
                    u *= v;//this step can be avoided (TODO optimize)
                }
                return ud;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        private static double modulo(double[] input, int j, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double result = Variable.get(0).activate(input, j);
                for (int Vi = 1; Vi < Variable.size(); Vi++) {
                    final double current = Variable.get(Vi).activate(input, j);
                    result %= current;
                }
                return result;
            } else {
                return Variable.get(0).derive(input, d, j);// j ?
            }
        }

        @Contract(pure = true)
        private static double modulo(double[] input, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double result = Variable.get(0).activate(input);
                for (int Vi = 1; Vi < Variable.size(); Vi++) {
                    final double current = Variable.get(Vi).activate(input);
                    result %= current;
                }
                return result;
            } else {
                return Variable.get(0).derive(input, d);
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        private static double subtraction(double[] input, int j, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double result = Variable.get(0).activate(input, j);
                for (int Vi = 1; Vi < Variable.size(); Vi++) {
                    final double current = Variable.get(Vi).activate(input, j);
                    result -= current;
                }
                return result;
            } else {
                double derivative = 0;
                for (int i = 0; i < Variable.size(); ++i) {
                    if (i == 0) {
                        derivative += Variable.get(i).derive(input, d, j);
                    } else {
                        derivative -= Variable.get(i).derive(input, d, j);
                    }
                }
                return derivative;
            }
        }

        @Contract(pure = true)
        private static double subtraction(double[] input, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double result = Variable.get(0).activate(input);
                for (int Vi = 1; Vi < Variable.size(); Vi++) {
                    final double current = Variable.get(Vi).activate(input);
                    result -= current;
                }
                return result;
            } else {
                double derivative = 0;
                for (int i = 0; i < Variable.size(); ++i) {
                    if (i == 0) {
                        derivative += Variable.get(i).derive(input, d);
                    } else {
                        derivative -= Variable.get(i).derive(input, d);
                    }
                }
                return derivative;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        @Contract(pure = true)
        private static double addition(double[] input, int j, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double result = Variable.get(0).activate(input, j);
                for (int Vi = 1; Vi < Variable.size(); Vi++) {
                    final double current = Variable.get(Vi).activate(input, j);
                    result += current;
                }
                return result;
            } else {
                double derivative = 0;
                for (int i = 0; i < Variable.size(); ++i) {
                    derivative += Variable.get(i).derive(input, d, j);
                }
                return derivative;
            }
        }

        @Contract(pure = true)
        private static double addition(double[] input, int d, ArrayList<IFunction> Variable) {
            if (d < 0) {
                double result = Variable.get(0).activate(input);
                for (int Vi = 1; Vi < Variable.size(); Vi++) {
                    final double current = Variable.get(Vi).activate(input);
                    result += current;
                }
                return result;
            } else {
                double derivative = 0;
                for (int i = 0; i < Variable.size(); ++i) {
                    derivative += Variable.get(i).derive(input, d);
                }
                return derivative;
            }
        }

        @Contract(pure = true)
        public static Tsr multiplication(Tsr tensor1, Tsr tensor2) {
            Tsr drn = new Tsr(tensor1.shape());
            int[] index = new int[drn.shape().length];
            int size = drn.size();
            for (int i = 0; i < size; i++) {
                Tsr.factory.io.addInto(drn, index, Tsr.factory.io.getFrom(tensor1, index) * Tsr.factory.io.getFrom(tensor2, index));
                Tsr.factory.util.increment(index, drn.shape());
            }
            return drn;
        }

        @Contract(pure = true)
        public static Tsr addition(Tsr tensor1, Tsr tensor2) {
            Tsr drn = new Tsr(tensor1.shape());
            int[] index = new int[drn.shape().length];
            int size = drn.size();
            for (int i = 0; i < size; i++) {
                Tsr.factory.io.addInto(drn, index, Tsr.factory.io.getFrom(tensor1, index) + Tsr.factory.io.getFrom(tensor2, index));
                Tsr.factory.util.increment(index, drn.shape());
            }
            return drn;
        }

        @Contract(pure = true)
        public static Tsr convection(Tsr tensor1, Tsr tensor2) {
            tensor1.setIsVirtual(false);
            tensor2.setIsVirtual(false);
            Tsr newTensor = new Tsr(Tsr.factory.util.shpOfCon(tensor1.shape(), tensor2.shape()));
            exec.convection(newTensor, tensor1, tensor2);
            return newTensor;
        }

        @Contract(pure = true)
        public static Tsr convection_inv(Tsr drain, Tsr source1, Tsr source2, boolean first) {
            source1.setIsVirtual(false);
            source2.setIsVirtual(false);
            drain.setIsVirtual(false);
            exec.convection_inv(source2, (!first) ? source1 : drain, (!first) ? drain : source1);
            return (first) ? source1 : source2;
        }

        @Contract(pure = true)
        public static void convection(Tsr t0_drain, Tsr t1_source, Tsr t2_source) {
            int[] t0Shp = t0_drain.shape();
            int[] t1Shp = t1_source.shape();
            int[] t2Shp = t2_source.shape();
            int[] t0Tln = t0_drain.translation();
            int[] t1Tln = t1_source.translation();
            int[] t2Tln = t2_source.translation();
            int rank = t0Shp.length;
            int[] t0Idx = new int[rank];
            int[] t1Idx = new int[rank];
            int[] t2Idx = new int[rank];
            double[] t0_value = t0_drain.targetValue();
            double[] t1_value = t1_source.targetValue();
            double[] t2_value = t2_source.targetValue();

            int drnSze = t0_drain.size();
            int i = 0;
            while (i < drnSze) {//increment on drain accordingly:
                int ri = 0;
                while (ri < rank) {
                    if (t1Shp[ri] == t2Shp[ri]) {//setting 0
                        t1Idx[ri] = t0Idx[ri];//mtch[mi];
                        t2Idx[ri] = t0Idx[ri];//mtch[mi];
                    } else if (t1Shp[ri] > t2Shp[ri]) {//setting hdr1 idx to id idx
                        t1Idx[ri] = t0Idx[ri];//mtch[mi];
                        t2Idx[ri] = 0;
                    } else if (t1Shp[ri] < t2Shp[ri]) {//setting hdr2 idx to id idx
                        t1Idx[ri] = 0;
                        t2Idx[ri] = t0Idx[ri];//mtch[mi];
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
                    if (incrementing == false) {
                        int i1 = Tsr.factory.util.iOf(t1Idx, t1Tln);
                        int i2 = Tsr.factory.util.iOf(t2Idx, t2Tln);
                        value += t1_value[i1] * t2_value[i2];
                        incrementing = true;
                        ri = 0;
                    } else {//incrementing:
                        if (t1Idx[ri] < t1Shp[ri] && t2Idx[ri] < t2Shp[ri]) {
                            t1Idx[ri]++;
                            t2Idx[ri]++;
                            if (t1Idx[ri] == t1Shp[ri] || t2Idx[ri] == t2Shp[ri]) {
                                if (ri == (rank - 1)) {
                                    running = false;
                                }
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
                                ri = 0;
                            }
                        } else {
                            ri++;
                        }
                    }
                }//setInto _value in drn:
                int i0 = Tsr.factory.util.iOf(t0Idx, t0Tln);
                t0_value[i0] = value;
                //System.out.println(i0 + " - " + i);
                i++;//increment on drain:
                if (i < drnSze) {
                    Tsr.factory.util.increment(t0Idx, t0Shp);
                }
            }
        }

        @Contract(pure = true)
        public static void convection_inv(Tsr t0_origin, Tsr t1_handle, Tsr t2_drain) {
            int[] t0Shp = t0_origin.shape();
            int[] t1Shp = t1_handle.shape();
            int[] t2Shp = t2_drain.shape();
            int[] t0Tln = t0_origin.translation();
            int[] t1Tln = t1_handle.translation();
            int[] t2Tln = t2_drain.translation();
            int rank = t0Shp.length;
            int[] t0Idx = new int[rank];
            int[] t1Idx = new int[rank];
            int[] t2Idx = new int[rank];
            double[] t0_value = t0_origin.targetValue();
            double[] t1_value = t1_handle.targetValue();
            double[] t2_value = t2_drain.targetValue();

            int drnSze = t0_origin.size();
            int i = 0;
            while (i < drnSze) {//increment on drain accordingly:
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
                    if (incrementing == false) {

                        boolean isMatch = true;
                        for (int rii = 0; rii < rank; rii++) {
                            if (!(t1Idx[rii] < t1Shp[rii] && t1Idx[rii] >= 0)) {
                                isMatch = false;
                            }
                        }
                        if (isMatch) {
                            int i1 = Tsr.factory.util.iOf(t1Idx, t1Tln);
                            int i2 = Tsr.factory.util.iOf(t2Idx, t2Tln);
                            value += t1_value[i1] * t2_value[i2];
                        }
                        incrementing = true;
                        ri = 0;
                    } else {//incrementing:
                        if (t2Idx[ri] < t2Shp[ri]) {
                            t2Idx[ri]++;
                            if (t2Idx[ri] == t2Shp[ri]) {
                                if (ri == (rank - 1)) {
                                    running = false;
                                }
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
                                ri = 0;
                            }
                        } else {
                            ri++;
                        }
                    }
                }
                //setInto _value in drn:
                int i0 = Tsr.factory.util.iOf(t0Idx, t0Tln);
                t0_value[i0] = value;
                i++;//increment on drain:
                if (i < drnSze) {
                    Tsr.factory.util.increment(t0Idx, t0Shp);
                }
            }
        }

    }

}
