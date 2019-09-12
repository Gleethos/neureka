package neureka.core.function.factory;

import neureka.core.T;
import neureka.core.function.IFunction;
import neureka.core.function.factory.autograd.GraphLock;
import neureka.core.function.factory.assembly.FunctionGraphBuilder;
import neureka.core.function.factory.implementations.FConstant;
import neureka.core.function.factory.autograd.GraphBuilder;
import neureka.core.device.Device;

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
        return IFunction.REGISTER[_id];
    }

    @Override
    public IFunction newBuild(String expression) {
        return FunctionGraphBuilder.newBuild(expression, true);
    }

    @Override
    public String toString() {
        String reconstructed = "";
        if (_source.size() == 1 && IFunction.REGISTER[_id].length() > 1) {
            String expression = _source.get(0).toString();
            if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                return IFunction.REGISTER[_id] + expression;
            }
            return IFunction.REGISTER[_id] + "(" + expression + ")";
        }
        reconstructed = ((IFunction.REGISTER[_id] == ",") ? "[" : "") + reconstructed;
        for (int i = 0; i < _source.size(); ++i) {
            if (_source.get(i) != null) {
                if ((IFunction.REGISTER[_id] == ",")) {
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
            if (i < _source.size() - ((IFunction.REGISTER[_id] == ",") ? 2 : 1)) {
                reconstructed = reconstructed + IFunction.REGISTER[_id];
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
    protected T tensorActivationOf(T input, boolean derive) {
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
            Calculation.foreach(input, output, (i, inputValue, outputValue) -> {
                outputValue[i] = scalarActivationOf(inputValue[i], derive);
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
    protected T tensorActivationOf(T[] input, int j, int d) {
        /**  The code below deals with deep functions (non flat):  * */
        if (d < 0 && !_isFlat) {//only flat functions can be executed
            if (_id <= 9) {
                return (new FunctionGraphBuilder().newBuild(IFunction.REGISTER[_id] + "(I[" + ((j < 0) ? 0 : j) + "])", true).activate(input));
            } else {
                if (IFunction.REGISTER[_id].length() != 1) {
                    /**  SUMMATION, PI,  * */
                    T[] tsrs = activateSource(input);
                    return (FunctionGraphBuilder.newBuild(IFunction.REGISTER[_id] + "(I[j])", true).activate(tsrs));
                } else if (_id <= 20) {
                    /**  '+', '-', 'x', '*', '%', '«', '»', ',', ...  * */
                    String operation = (IFunction.REGISTER[_id].length() > 1) ? IFunction.REGISTER[_id] : "";
                    T[] tsrs = activateSource(input, j, null);
                    for (int i = 0; i < tsrs.length; i++) {
                        operation += "I[" + i + "]" + ((i + 1 < tsrs.length) ? IFunction.REGISTER[_id] : "");
                    }
                    if (j < 0) {
                        return (FunctionGraphBuilder.newBuild(operation, _doAD).activate(tsrs));
                    } else {
                        return (FunctionGraphBuilder.newBuild(operation, _doAD).activate(tsrs, j));
                    }
                } else {
                    /**  Tensor shape translation: * */
                    T[] tsrs = activateSource(input, j, new int[]{1});
                    if (j < 0) {
                        return (FunctionGraphBuilder.newBuild(_id, tsrs.length, _doAD).activate(tsrs));
                    } else {
                        return (FunctionGraphBuilder.newBuild(_id, tsrs.length, _doAD).activate(tsrs, j));
                    }
                }
            }
        }
        /**  The following code is reached in flat functions only:  * */
        T output = execute(input, j, d);
        /**  Autograd-Graph will be generated below for the new GraphNode: **/
        if (d < 0 && _doAD) {
            GraphBuilder.connect(output, input, this);
        }
        return output;
    }

    private T execute(T[] input, int j, int d) {
        Device device = (Device) input[0].find(Device.class);
        boolean onSameDevice = T.factory.util.shareGuestDevice(input);
        if (onSameDevice && IFunction.REGISTER[_id] != "," &&
                !((IFunction.REGISTER[_id] == "x" || IFunction.REGISTER[_id] == "«" || IFunction.REGISTER[_id] == "»") && d > -1)
        ) {
            int[] shp = (IFunction.REGISTER[_id] == "x")?T.factory.util.shpOfCon(input[0].shape(), input[1].shape()):input[0].shape();
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
        } else {
            if (IFunction.REGISTER[_id] == "x") {
                if (d < 0) {
                    return T.factory.exec.convolution(input[0], input[1]);
                } else {
                    if (d == 0) {
                        return (input[1]);
                    } else {
                        return (input[0]);
                    }
                }
            } else if (IFunction.REGISTER[_id] == "" + ((char) 171) || IFunction.REGISTER[_id] == "" + ((char) 187)) {
                if (d < 0) {//  ""+((char)171), ""+((char)187) //<< / >>
                    if (IFunction.REGISTER[_id] == "" + ((char) 187)) {
                        return T.factory.exec.convolution_inv(input[0], input[1], input[2], false);
                    } else {
                        return T.factory.exec.convolution_inv(input[2], input[1], input[0], false);
                    }
                } else {//Todo: What then? :
                    if (d == 0) {
                        return input[1];
                    } else {
                        return input[0];
                    }
                }
            } else if (IFunction.REGISTER[_id] == ",") {
                int[] newForm = new int[input.length - 1];
                for (int i = 0; i < input.length - 1; i++) {
                    newForm[i] = (int) T.factory.io.getFrom(input[i], 0);
                }
                if (d < 0) {
                    T t = input[input.length - 1];
                    return T.factory.exec.reshaped(t, newForm, true);//t.reshape(newForm);

                } else {//reverse reshape:
                    /**
                     *      [3, 2, 4, 0, 1]
                     *      [0, 1, 2, 3, 4]
                     * */
                    int[] reversed = new int[newForm.length];
                    for (int i = 0; i < newForm.length; i++) {
                        // reversed[newForm[i]] = i;
                    }
                }
            } else {
                T[] tsrs = input;
                double[] inp = new double[tsrs.length];
                T output = T.factory.newTensor(input[0].shape(), input[0].translation());
                T finalOutput = output;
                output.foreach((i) -> {
                    for (int ii = 0; ii < tsrs.length; ii++) {
                        inp[ii] = tsrs[ii].value()[i];
                    }

                    finalOutput.value()[i] = scalarActivationOf(inp, j, d);
                });
                return  output;
            }
        }
        //Todo: warning/exception.....
        return T.factory.newTensor(input[0].shape(), input[0].translation());
    }

    private T[] activateSource(T[] input) {
        T[] tsrs = new T[input.length];
        for (int i = 0; i < tsrs.length; i++) {
            tsrs[i] = _source.get(0).activate(input, i);
        }
        return tsrs;
    }

    private T[] activateSource(T[] input, int j, int[] templateShape) {
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
    protected double scalarActivationOf(double input, boolean derive) {
        switch (_id) {
            case 0:
                return Calculation.getReLuOf(input, derive);
            case 1:
                return Calculation.getSigmoidOf(input, derive);
            case 2:
                return Calculation.getTanhOf(input, derive);
            case 3:
                return Calculation.getQuadraticOf(input, derive);
            case 4:
                return Calculation.getLigmoidOf(input, derive);
            case 5:
                return Calculation.getLinearOf(input, derive);
            case 6:
                return Calculation.getGaussianOf(input, derive);
            case 7:
                return Calculation.getAbsoluteOf(input, derive);
            case 8:
                return Calculation.getSinusOf(input, derive);
            case 9:
                return Calculation.getCosinusOf(input, derive);
            default:
                return input;
        }
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    protected double scalarActivationOf(double[] input, int j, int d) {
        switch (_id) {
            case 10:
                return (j < 0) ? Calculation.getSummation(input, d, _source) : Calculation.getSummation(input, j, d, _source);
            case 11:
                return (j < 0) ? Calculation.getPI(input, d, _source) : Calculation.getPI(input, j, d, _source);
            case 12:
                return (j < 0) ? Calculation.getPowerOf(input, d, _source) : Calculation.getPowerOf(input, j, d, _source);
            case 13:
                return (j < 0) ? Calculation.getDivision(input, d, _source) : Calculation.getDivision(input, j, d, _source);
            case 14:
                return (j < 0) ? Calculation.getMultiplication(input, d, _source) : Calculation.getMultiplication(input, j, d, _source);
            case 15:
                return (j < 0) ? Calculation.getModulo(input, d, _source) : Calculation.getModulo(input, j, d, _source);
            case 16:
                return (j < 0) ? Calculation.getSubtraction(input, d, _source) : Calculation.getSubtraction(input, j, d, _source);
            case 17:
                return (j < 0) ? Calculation.getAddition(input, d, _source) : Calculation.getAddition(input, j, d, _source);
            case 18:
                return (j < 0) ? Calculation.getMultiplication(input, d, _source) : Calculation.getMultiplication(input, j, d, _source);
            default:
                return 0;
        }
    }

    private static class Calculation {
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
        public static double getReLuOf(double input, boolean derive) {
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
        public static double getSigmoidOf(double input, boolean derive) {
            if (!derive) {
                return 1 / (1 + Math.pow(Math.E, (-input)));
            } else {
                return (Math.pow(Math.E, -(input))) / (Math.pow((1 + Math.pow(Math.E, -(input))), 2) + 2 * Math.pow(Math.E, -(input)));
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getTanhOf(double input, boolean derive) {
            if (!derive) {
                return ((input)) / Math.pow((1 + Math.pow(((input)), 2)), 0.5);
            } else {
                return (1 - Math.pow(((input) / Math.pow((1 + Math.pow((input), 2)), 0.5)), 2));
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getQuadraticOf(double input, boolean derive) {
            if (!derive) {
                return ((input) * (input));
            } else {
                return 2 * input;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getLigmoidOf(double input, boolean derive) {
            if (!derive) {
                return (Math.log(1+Math.pow(Math.E, input)));
            } else {
                return getSigmoidOf(input, false);
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getLinearOf(double input, boolean derive) {
            if (!derive) {
                return (input);
            } else {
                return 1;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getGaussianOf(double input, boolean derive) {
            if (!derive) {
                return Math.pow(Math.E, -Math.pow((input), 2));
            } else {
                return -2 * ((input)) * Math.pow(Math.E, -Math.pow((input), 2));
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getAbsoluteOf(double input, boolean derive) {
            if (!derive) {
                return Math.abs(input);
            } else {
                return (input < 0) ? -1 : 1;
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getSinusOf(double input, boolean derive) {
            if (!derive) {
                return Math.sin(input);
            } else {
                return Math.cos(input);
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getCosinusOf(double input, boolean derive) {
            if (!derive) {
                return Math.cos(input);
            } else {
                return -Math.sin(input);
            }
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        private static double getSummation(double[] input, int j, int d, ArrayList<IFunction> Variable) {
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

        private static double getSummation(double[] input, int d, ArrayList<IFunction> Variable) {
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
        private static double getPI(double[] input, int j, int d, ArrayList<IFunction> Variable) {
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

        private static double getPI(double[] input, int d, ArrayList<IFunction> Variable) {
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
        private static double getPowerOf(double[] input, int j, int d, ArrayList<IFunction> Variable) {
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

        private static double getPowerOf(double[] input, int d, ArrayList<IFunction> Variable) {
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
        private static double getDivision(double[] input, int j, int d, ArrayList<IFunction> Variable) {
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

        private static double getDivision(double[] input, int d, ArrayList<IFunction> Variable) {
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
        private static double getMultiplication(double[] input, int j, int d, ArrayList<IFunction> Variable) {
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

        private static double getMultiplication(double[] input, int d, ArrayList<IFunction> Variable) {
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
        private static double getModulo(double[] input, int j, int d, ArrayList<IFunction> Variable) {
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

        private static double getModulo(double[] input, int d, ArrayList<IFunction> Variable) {
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
        private static double getSubtraction(double[] input, int j, int d, ArrayList<IFunction> Variable) {
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

        private static double getSubtraction(double[] input, int d, ArrayList<IFunction> Variable) {
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
        private static double getAddition(double[] input, int j, int d, ArrayList<IFunction> Variable) {
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

        private static double getAddition(double[] input, int d, ArrayList<IFunction> Variable) {
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
    }

}
