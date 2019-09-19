package neureka.core.function.factory;

import neureka.core.T;
import neureka.core.function.IFunction;
import neureka.core.function.factory.autograd.GraphLock;
import neureka.core.function.factory.assembly.FunctionGraphBuilder;
import neureka.core.function.factory.implementations.FConstant;
import neureka.core.function.factory.autograd.GraphBuilder;
import neureka.core.device.Device;
import org.jetbrains.annotations.Contract;

import java.util.ArrayList;

public abstract class Function implements IFunction {

    protected int _id;
    protected boolean _isFlat;
    protected boolean _doAD;
    protected ArrayList<IFunction> _source;

    /**
     * @param f_id
     * @param isFlat
     * @param Srcs
     * @param doAD
     */
    protected Function(int f_id, boolean isFlat, ArrayList<IFunction> Srcs, boolean doAD) {
        _id = f_id;
        _isFlat = isFlat;
        _source = Srcs;
        _doAD = doAD;
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
    public IFunction newBuild(String expression) {
        return FunctionGraphBuilder.newBuild(expression, true);
    }

    @Override
    public String toString() {
        String reconstructed = "";
        if (_source.size() == 1 && TYPES.REGISTER[_id].length() > 1) {
            String expression = _source.get(0).toString();
            if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                return TYPES.REGISTER[_id] + expression;
            }
            return TYPES.REGISTER[_id] + "(" + expression + ")";
        } else {
            reconstructed = ((TYPES.REGISTER[_id] == ",") ? "[" : "") + reconstructed;
            for (int i = 0; i < _source.size(); ++i) {
                if (_source.get(i) != null) {
                    if ((TYPES.REGISTER[_id] == ",")) {
                        if (i == _source.size() - 1) {
                            reconstructed = reconstructed
                                    + "]:(" + (
                                    (_source.get(i) instanceof FConstant)
                                            ? _source.get(i).toString().split("\\.")[0]
                                            : _source.get(i).toString()
                            ) + ")";
                        } else {
                            reconstructed = reconstructed
                                    + (
                                    (_source.get(i) instanceof FConstant)
                                            ? _source.get(i).toString().split("\\.")[0]
                                            : _source.get(i).toString()
                            );
                        }
                    } else {
                        reconstructed = reconstructed + _source.get(i).toString();
                    }
                } else {
                    reconstructed = reconstructed + "(null)";
                }
                if (i < _source.size() - ((TYPES.REGISTER[_id] == ",") ? 2 : 1)) {
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
    public abstract T activate(T[] input, int j);

    @Override
    public abstract T activate(T[] input);

    @Override
    public abstract T derive(T[] input, int index, int j);

    @Override
    public abstract T derive(T[] input, int index);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public abstract double activate(final double[] input, int j);

    @Override
    public abstract double activate(final double[] input);

    @Override
    public abstract double derive(final double[] input, final int index, final int j);

    @Override
    public abstract double derive(final double[] input, final int index);

    //Activation stage 1: IFunction determination!
    //============================================================================================================================================================================================

    /**
     * Responsible for handling functions with id's 0-9  (single input functions!)
     */
    protected T _tensor_activation(T input, boolean derive) {
        T output = T.factory.newTensor(input.shape(), input.translation());
        if (!derive && !_isFlat) {
            output.inject(new FunctionGraphBuilder().newBuild(_id, 1, true).activate(new T[]{input}));
            output.add(input.find(GraphLock.class));
            return output;
        }
        if (input.isOutsourced()) {
            Device device = (Device) input.find(Device.class);
            device.add(output);
            device.calculate(new T[]{output, input}, _id, (derive) ? 0 : -1);
        } else {
            exec.foreach(input, output, (i, inputValue, outputValue) -> {
                outputValue[i] = _scalar_activation(inputValue[i], derive);
            });
        }
        if (!derive && _doAD) {
            GraphBuilder.connect(output, new T[]{input}, this);
        }
        output.add(input.find(GraphLock.class));
        return output;
    }

    /**
     * Responsible for handling functions with multiple inputs!
     *
     * @param input
     * @param j
     * @param d
     * @return
     */
    protected T _tensor_activation(T[] input, int j, int d) {
        /**  The code below deals with deep functions (non flat):  * */
        if (d < 0 && !_isFlat) {//only flat functions can be executed
            if (TYPES.isFunction(_id)) {
                return (new FunctionGraphBuilder().newBuild(TYPES.REGISTER[_id] + "(I[" + ((j < 0) ? 0 : j) + "])", true).activate(input));
            } else {
                if (TYPES.isFunction(_id)||TYPES.isIndexer(_id)) {
                    /**  SUMMATION, PI,  * */
                    T[] tsrs = _source_activation(input);
                    return (FunctionGraphBuilder.newBuild(TYPES.REGISTER[_id] + "(I[j])", true).activate(tsrs));
                } else if (TYPES.isOperation(_id)) {
                    /**  '+', '-', 'x', '*', '%', '«', '»', ',', ...  * */
                    String operation = (TYPES.REGISTER[_id].length() > 1) ? TYPES.REGISTER[_id] : "";
                    T[] tsrs = _source_activation(input, j, null);
                    for (int i = 0; i < tsrs.length; i++) {
                        operation += "I[" + i + "]" + ((i + 1 < tsrs.length) ? TYPES.REGISTER[_id] : "");
                    }
                    if (j < 0) {
                        return (FunctionGraphBuilder.newBuild(operation, _doAD).activate(tsrs));
                    } else {
                        return (FunctionGraphBuilder.newBuild(operation, _doAD).activate(tsrs, j));
                    }
                } else {
                    /**  Tensor shape translation: * */
                    T[] tsrs = _source_activation(input, j, new int[]{1});
                    if (j < 0) {
                        return (FunctionGraphBuilder.newBuild(_id, tsrs.length, _doAD).activate(tsrs));
                    } else {
                        return (FunctionGraphBuilder.newBuild(_id, tsrs.length, _doAD).activate(tsrs, j));
                    }
                }
            }
        }
        /**  The following code is reached in flat functions only:  * */
        T output = _execute(input, j, d);
        /**  Autograd-Graph will be generated below for the new GraphNode: **/
        if (d < 0 && _doAD) {
            GraphBuilder.connect(output, input, this);
        }
        return output;
    }

    private T _execute(T[] input, int j, int d) {
        Device device = (Device) input[0].find(Device.class);
        boolean onSameDevice = T.factory.util.shareGuestDevice(input);
        if (onSameDevice && TYPES.REGISTER[_id] != "," && (!TYPES.isConvection(_id) && d > -1)) {
            if(TYPES.REGISTER[_id]=="<") {
                device.overwrite(_source.get(0).activate(input), _source.get(1).activate(input));
            } else if(TYPES.REGISTER[_id]==">") {
                device.overwrite(_source.get(1).activate(input), _source.get(0).activate(input));
            } else {
                int[] shp = (TYPES.REGISTER[_id] == "x")?T.factory.util.shpOfCon(input[0].shape(), input[1].shape()):input[0].shape();
                T output = new T(shp, 0.0);
                if (device != null) {
                    device.add(output);
                }
                for (int i = 0; i < input.length; i++) {
                    device = (Device) input[i].find(Device.class);
                    T[] tsrs = new T[1 + input.length];
                    tsrs[0] = output;
                    for (int ii = 1; ii < tsrs.length; ii++) {
                        tsrs[ii] = input[ii - 1];
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
                }
                return output;
            }
        } else {
            if (TYPES.REGISTER[_id] == "x") {
                if (d < 0) {
                    return exec.convolution(input[0], input[1]);
                } else {
                    if (d == 0) {
                        return (input[1]);
                    } else {
                        return (input[0]);
                    }
                }
            } else if (TYPES.REGISTER[_id] == "" + ((char) 171) || TYPES.REGISTER[_id] == "" + ((char) 187)) {
                if (d < 0) {//  ""+((char)171), ""+((char)187) //<< / >>
                    if (TYPES.REGISTER[_id] == "" + ((char) 187)) {
                        return exec.convolution_inv(input[0], input[1], input[2], false);
                    } else {
                        return exec.convolution_inv(input[2], input[1], input[0], false);
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
                    newForm[i] = (int) T.factory.io.getFrom(input[i], 0);
                }
                if (d < 0) {
                    T t = input[input.length - 1];
                    return T.factory.exec.reshaped(t, newForm, true);//t.reshape(newForm);
                } else {//reverse reshape:
                    /**      [3, 2, 4, 0, 1]
                     *      [0, 1, 2, 3, 4]
                     * */
                    int[] reversed = new int[newForm.length];
                    for (int i = 0; i < newForm.length; i++) {
                        if(newForm[i]>=0){
                            reversed[newForm[i]] = i;
                        }
                    }
                }
            } else if(TYPES.REGISTER[_id]=="<") {
                return _source.get(0).activate(input).setTargetValue(_source.get(1).activate(input).targetValue(true));
            } else if(TYPES.REGISTER[_id]==">") {
                return _source.get(1).activate(input).setTargetValue(_source.get(0).activate(input).targetValue(true));
            } else {
                T[] tsrs = input;
                double[] inp = new double[tsrs.length];
                T output = T.factory.newTensor(input[0].shape(), input[0].translation());
                T finalOutput = output;
                output.foreach((i) -> {
                    for (int ii = 0; ii < tsrs.length; ii++) {
                        inp[ii] = tsrs[ii].value()[i];
                    }

                    finalOutput.value()[i] = _scalar_activation(inp, j, d);
                });
                return  output;
            }
        }
        //Todo: warning/exception.....
        return T.factory.newTensor(input[0].shape(), input[0].translation());
    }

    private T[] _source_activation(T[] input) {
        T[] tsrs = new T[input.length];
        for (int i = 0; i < tsrs.length; i++) {
            tsrs[i] = _source.get(0).activate(input, i);
        }
        return tsrs;
    }

    private T[] _source_activation(T[] input, int j, int[] templateShape) {
        T[] tsrs = new T[_source.size()];
        for (int i = 0; i < tsrs.length; i++) {//constants need to be figured out!
            if (_source.get(i) instanceof FConstant) {
                tsrs[i] = null;
            } else {
                tsrs[i] =
                        (j < 0)
                                ? _source.get(i).activate(input)
                                : _source.get(i).activate(input, j);
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
                            ? T.factory.newTensor(((FConstant) _source.get(i)).value(), templateShape)
                            : T.factory.newTensor(_source.get(i).activate(new double[]{}, j), templateShape);
        }
        return tsrs;
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
                return (j < 0) ? exec.summation(input, d, _source) : exec.summation(input, j, d, _source);
            case 11:
                return (j < 0) ? exec.PI(input, d, _source) : exec.PI(input, j, d, _source);
            case 12:
                return (j < 0) ? exec.power(input, d, _source) : exec.power(input, j, d, _source);
            case 13:
                return (j < 0) ? exec.division(input, d, _source) : exec.division(input, j, d, _source);
            case 14:
                return (j < 0) ? exec.multiplication(input, d, _source) : exec.multiplication(input, j, d, _source);
            case 15:
                return (j < 0) ? exec.modulo(input, d, _source) : exec.modulo(input, j, d, _source);
            case 16:
                return (j < 0) ? exec.subtraction(input, d, _source) : exec.subtraction(input, j, d, _source);
            case 17:
                return (j < 0) ? exec.addition(input, d, _source) : exec.addition(input, j, d, _source);
            case 18:
                return (j < 0) ? exec.multiplication(input, d, _source) : exec.multiplication(input, j, d, _source);
            default:
                return 0;
        }
    }

    public static class exec {
        private interface Actor {
            void apply(Integer i, double[] v1, double[] v2);
        }

        private static void foreach(T t1, T t2, Actor action) {
            double[] inputValue = (t1.value() == null) ? new double[t1.size()] : t1.value();
            double[] outputValue = (t2.value() == null) ? new double[t2.size()] : t2.value();
            t1.foreach((i) -> {
                action.apply(i, inputValue, outputValue);
            });
            t2.setValue(outputValue);
            t1.setValue(inputValue);
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        //Activation stage 1: Activation _id determination
        //============================================================================================================================================================================================
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
        public static double sigmoid(double input, boolean derive) {
            if (!derive) {
                return 1 / (1 + Math.pow(Math.E, (-input)));
            } else {
                return (Math.pow(Math.E, -(input))) / (Math.pow((1 + Math.pow(Math.E, -(input))), 2) + 2 * Math.pow(Math.E, -(input)));
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double tanh(double input, boolean derive) {
            if (!derive) {
                return ((input)) / Math.pow((1 + Math.pow(((input)), 2)), 0.5);
            } else {
                return (1 - Math.pow(((input) / Math.pow((1 + Math.pow((input), 2)), 0.5)), 2));
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double quadratic(double input, boolean derive) {
            if (!derive) {
                return ((input) * (input));
            } else {
                return 2 * input;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double ligmoid(double input, boolean derive) {
            if (!derive) {
                return (Math.log(1+Math.pow(Math.E, input)));
            } else {
                return sigmoid(input, false);
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double linear(double input, boolean derive) {
            if (!derive) {
                return (input);
            } else {
                return 1;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double gaussian(double input, boolean derive) {
            if (!derive) {
                return Math.pow(Math.E, -Math.pow((input), 2));
            } else {
                return -2 * ((input)) * Math.pow(Math.E, -Math.pow((input), 2));
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double absolute(double input, boolean derive) {
            if (!derive) {
                return Math.abs(input);
            } else {
                return (input < 0) ? -1 : 1;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double sinus(double input, boolean derive) {
            if (!derive) {
                return Math.sin(input);
            } else {
                return Math.cos(input);
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double cosinus(double input, boolean derive) {
            if (!derive) {
                return Math.cos(input);
            } else {
                return -Math.sin(input);
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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



        public static T convolution(T tensor1, T tensor2) {
            tensor1.setIsVirtual(false);
            tensor2.setIsVirtual(false);
            T newTensor = new T(T.factory.util.shpOfCon(tensor1.shape(), tensor2.shape()));
            exec.tensMul(newTensor, tensor1, tensor2);
            return newTensor;
        }

        public static T convolution_inv(T drain, T source1, T source2, boolean first) {
            source1.setIsVirtual(false);
            source2.setIsVirtual(false);
            drain.setIsVirtual(false);
            exec.tensMul_inv(source2, (!first) ? source1 : drain, (!first) ? drain : source1);
            return (first) ? source1 : source2;
        }

        public static T multiplication(T tensor1, T tensor2) {
            T drn = new T(tensor1.shape());
            int[] index = new int[drn.shape().length];
            int size = drn.size();
            for (int i = 0; i < size; i++) {
                T.factory.io.addInto(drn, index, T.factory.io.getFrom(tensor1, index) * T.factory.io.getFrom(tensor2, index));
                T.factory.util.increment(index, drn.shape());
            }
            return drn;
        }

        public static T addition(T tensor1, T tensor2) {
            T drn = new T(tensor1.shape());
            int[] index = new int[drn.shape().length];
            int size = drn.size();
            for (int i = 0; i < size; i++) {
                T.factory.io.addInto(drn, index, T.factory.io.getFrom(tensor1, index) + T.factory.io.getFrom(tensor2, index));
                T.factory.util.increment(index, drn.shape());
            }
            return drn;
        }

        @Contract(pure = true)
        public static void tensMul(T t0_drain, T t1_source, T t2_source) {
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
            double[] t0_value = (t0_drain.gradientIsTargeted())?t0_drain.gradient():t0_drain.value();
            double[] t1_value = (t1_source.gradientIsTargeted())?t1_source.gradient():t1_source.value();
            double[] t2_value = (t2_source.gradientIsTargeted())?t2_source.gradient():t2_source.value();

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
                        int i1 = T.factory.util.iOf(t1Idx, t1Tln);
                        int i2 = T.factory.util.iOf(t2Idx, t2Tln);
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
                int i0 = T.factory.util.iOf(t0Idx, t0Tln);
                t0_value[i0] = value;
                //System.out.println(i0 + " - " + i);
                i++;//increment on drain:
                if (i < drnSze) {
                    T.factory.util.increment(t0Idx, t0Shp);
                }
            }
        }

        @Contract(pure = true)
        public static void tensMul_inv(T t0_origin, T t1_handle, T t2_drain) {
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
            double[] t0_value = (t0_origin.gradientIsTargeted())?t0_origin.gradient():t0_origin.value();
            double[] t1_value = (t1_handle.gradientIsTargeted())?t1_handle.gradient():t1_handle.value();
            double[] t2_value = (t2_drain.gradientIsTargeted())?t2_drain.gradient():t2_drain.value();


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
                            int i1 = T.factory.util.iOf(t1Idx, t1Tln);
                            int i2 = T.factory.util.iOf(t2Idx, t2Tln);
                            value += t1_value[i1] * t2_value[i2];
                            //1*-2 +2*3 -3*6 +2*3, 1*3 +2*6 -3*3 +2*-1,
                            //1*0  +2*2 -3*4 +2*2  +  4*-2 -2*3 -1*6 +5*3, 1*2 +2*4 -3*2 +2*1  +  4*3 -2*6 -1*3 +5*-1,
                            //4*0  -2*2 -1*4 +5*2, 4*2 -2*4 -1*2 +5*1
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
                int i0 = T.factory.util.iOf(t0Idx, t0Tln);
                t0_value[i0] = value;
                i++;//increment on drain:
                if (i < drnSze) {
                    T.factory.util.increment(t0Idx, t0Shp);
                }
            }
        }

    }

}
