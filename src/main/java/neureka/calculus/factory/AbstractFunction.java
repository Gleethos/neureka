package neureka.calculus.factory;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;
import neureka.calculus.factory.assembly.FunctionBuilder;
import neureka.autograd.GraphNode;
import neureka.calculus.factory.components.FunctionConstant;
import org.jetbrains.annotations.Contract;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import neureka.calculus.environment.implementations.*;

public abstract class AbstractFunction extends BaseFunction {
    
    private final OperationType _type;
    private final boolean _isFlat;
    private final boolean _doAD;
    private final ArrayList<Function> _src;

    //------------------------------------------------------------------------------------------------------------------

    /**
     * @param f_id
     * @param isFlat
     * @param source
     * @param doAD
     */
    protected AbstractFunction(int f_id, boolean isFlat, ArrayList<Function> source, boolean doAD) {
        _type = OperationType.instance(f_id);
        _isFlat = isFlat;
        _src = source;
        _doAD = doAD;
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public Function newBuild(String expression) {
        return FunctionBuilder.build(expression, true);
    }

    @Override
    public boolean isFlat() {
        return _isFlat;
    }

    @Override
    public boolean doesAD(){
        return _doAD;
    }

    @Override
    public int id() {
        return _type.id();
    }

    @Override
    public OperationType type() {
        return _type;
    }

    //---

    @Override
    public String toString()
    {
        List<String> stringedSource = _src.stream().map(e->((e==null)?"(null)":e.toString())).collect(Collectors.toList());
        String asStr = _type.getStringifier().asString(stringedSource);
        return asStr;
    }

    @Override
    public boolean dependsOn(int index) {
        for (Function f : _src) if (f.dependsOn(index)) return true;
        return false;
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public ADAgent getADAgent(Tsr[] inputs, int i, boolean forward){
        return _type.getADAgentOf(this, inputs, i, forward);
    }

    /**
     * Responsible for handling functions with multiple inputs!
     *
     * @param inputs
     * @param j
     * @param d
     * @return
     */
    protected Tsr _tensor_activation( Tsr[] inputs, int j, int d )
    {
        Device device = _device( inputs );
        ExecutionCall call = new ExecutionCall( device, inputs, d, _type );

        /* The code below deals with deep functions (non flat):  */
        if ( _isFlat ) {
            /* The following code is reached in flat functions only:  */
            /* Autograd-Graph will be generated below for the new GraphNode: */
            /* only flat functions can be executed directly*/
            if ( d < 0 && _doAD )
                return new GraphNode(this, call, ()-> __flat_execution( call, j )).getPayload();
            else
                return __flat_execution( call, j );

        } else return _apply(device, d, () -> __deep_execution( call, j ));
    }

    private Tsr __flat_execution( ExecutionCall call, int j )
    {
        Device myDevice = call.getDevice();
        int d = call.getDerivativeIndex();
        Tsr[] inputs = call.getTensors();

        if (call.getType().identifier().equals("x"))
        {
            //TODO: Move ALL of this into the operation types!!!!!
            Tsr[] tsrs = new Tsr[]{ null, inputs[0], inputs[1] };// _src_acti(inputs, j, -1, 1);
            tsrs[0] = ( d < 0 )
                    ? new Tsr(Tsr.Utility.Indexing.shpOfCon(tsrs[1].getNDConf().shape(), tsrs[2].getNDConf().shape()))
                    : null;

            for ( Tsr t : tsrs ) if( t != null ) t.setIsVirtual(false);
            myDevice.execute( new ExecutionCall( myDevice, tsrs, d, _type ) );
            return tsrs[0];

        } else if (
                _type.id() == OperationType.instance("<<x").id() ||
                        _type.id() == OperationType.instance("x>>").id()
        ) {
            if (d < 0) {
                //inputs = _src_acti(inputs, j, -1, 0);
                Tsr[] tsrs = _src_acti(inputs, j, -1, 0);
                for ( Tsr t : tsrs ) t.setIsVirtual(false);
                myDevice.execute( new ExecutionCall( myDevice, tsrs, 0, _type ) );
                if (_type.id() == OperationType.instance("x>>").id()) return tsrs[2];
                else return tsrs[0];
            }
            return null;
        } else if (_type.identifier().equals(",")) {
            inputs = _src_acti(inputs, j, -1, 0);
            int[] newForm = new int[_src.size() - 1];
            for (int i = 0; i < _src.size() - 1; i++) {
                newForm[i] = (int) Tsr.IO.getFrom(inputs[i], 0);//_src.get(i).call(inputs)
            }
            if (d >= 0) {//reverse reshape:
                int reverseLength = 0;
                for (int e : newForm) {
                    if (e>=0) reverseLength++;
                }
                int[] reversed = new int[reverseLength];
                int reshape_i = 0;
                int reverse_i = 0;
                while ( reverse_i < reverseLength ) {
                    if ( newForm[ reshape_i ] >= 0 ) {
                        reversed[ newForm[reshape_i] ] = reshape_i;
                        reverse_i++;
                    }
                    reshape_i++;
                }
                newForm = reversed;
            }
            Tsr t = inputs[inputs.length - 1];
            return Tsr.Exec.reshaped(t, newForm, true);
        } else {
            Tsr[] tsrs = inputs;
            return _apply(myDevice, d, () -> __deep_execution( new ExecutionCall( myDevice, tsrs, d, call.getType() ), j ));
        }
    }

    private Tsr __deep_execution( ExecutionCall call, int j )
    {
        Tsr[] inputs = call.getTensors();
        int d = call.getDerivativeIndex();
        Device device = call.getDevice();

        Tsr[] tsrs;
        if ( _type.isIndexer() ) tsrs = new Tsr[ 1 + inputs.length ];
        else tsrs = new Tsr[1 + _src.size()];

        if ( d >= 0 ) // Differentiation
        {
            //Chain-rule (forward AutoDiff):
            //inner times outer means:
            //first derive source!
            //like so:
            if ( _type.isIndexer() ) {
                for (int i = 1; i < tsrs.length; i++) {
                    tsrs[i] = _src.get(0).derive(inputs, d, i - 1);
                }
            } else {
                for ( int i = 1; i < tsrs.length; i++ ) {
                    tsrs[i] = ( j >= 0 ) ? _src.get( i - 1 ).derive( inputs, d, j ) : _src.get( i - 1 ).derive(inputs, d);
                }
            }
            //then add them all together! (is possible because of linearity...)
            Tsr inner;
            if ( tsrs.length > 2 ) {// Optimization: Finds index of "1.0" among otherwise all "0.0" virtual tensors!
                int index = ___indexOfFoundDerivative(tsrs);
                if ( index >= 0 ) inner = tsrs[index];
                else {
                    // Optimization above did not apply, so we accumulate all the derivatives!
                    device.execute( new ExecutionCall( device, tsrs, -1, OperationType.instance("+") ) );
                    inner = tsrs[0];//this is now the inner derivative!
                }
            } else inner = tsrs[1];

            tsrs[0] = null;
            //...then activate (No differentiation!) the source like so:
            if ( _type.isIndexer() ) { // Indexer pass an index j of course!
                for ( int i = 1; i < tsrs.length; i++ ) {
                    tsrs[i] = _src.get(0).call(inputs, i - 1); // i - 1 := j
                }
            } else {
                for ( int i = 1; i < tsrs.length; i++ ) {
                    tsrs[i] = (j >= 0) ? _src.get(i - 1).call(inputs, j) : _src.get(i - 1).call(inputs);
                }
            }
            //get derivative index within src list:
            for ( int i = 0; i < _src.size(); i++ ) {
                if ( _src.get(i).dependsOn(d) && !_type.isIndexer() ) {
                    d = i;
                    break;
                }
            }
            //Use those tensors for the outer derivative:
            device.execute( new ExecutionCall( device, tsrs, d, _type ) );
            //At the end:
            //multiply inner times outer: ( if inner is not 1 entirely... )
            if ( !( ( inner.isVirtual() || inner.size()==1 ) && inner.value64(0)==1.0) ) {
                tsrs = new Tsr[]{null, inner, tsrs[0]};
                device.execute( new ExecutionCall( device, tsrs, -1, OperationType.instance("*") ) );
            } // done!
            return tsrs[0];
        }
        else // No differentiation :
        {
            if (_type.isIndexer()) {
                for (int i = 1; i < tsrs.length; i++) tsrs[i] = _src.get(0).call(inputs, i - 1);
            } else if (
                    !_isFlat && j < 0 && (
                            _type.isOperation() || _type.supportsImplementation(Activation.class)
                    )
            ) {/*  '+', '-', 'x', '*', '%', '«', '»', ',', ...  */
                tsrs = _src_acti(inputs, j, d, 0);
                List<String> stringedSource = IntStream.range(0, _src.size()).mapToObj(i -> "I[" + i + "]").collect(Collectors.toList());
                String asStr = _type.getStringifier().asString(stringedSource);
                return FunctionBuilder.build(asStr, _doAD).call(tsrs);
            } else {
                tsrs = _src_acti(inputs, j, d, 1);
            }
            device.execute( new ExecutionCall( device, tsrs, d, _type ) );
        }
        return (tsrs[0] == null) ? tsrs[1] : tsrs[0];
    }

    /**
     *  This method return the index of the tensor
     *  in the given tensor array which is virtual and contains "1.0".
     *  However if not all tensors are virtual or their values are not all "0.0" except one
     *  whose value is "1.0" then it return -1, because the optimization cannot
     *  be made...
     *
     * @param tsrs An array of tensors which ought to be analyzed.
     * @return The index of the tensor whose value is "1.0" (if all other are "0.0"), otherwise : -1
     */
    private int ___indexOfFoundDerivative(Tsr[] tsrs ) {
        boolean allVirtual = true;
        for ( Tsr t : tsrs ) if ( t != null && !t.isVirtual() ) allVirtual = false;
        if ( allVirtual ) {
            int index = -1;
            for ( int i=0; i < tsrs.length; i++ ) {
                double value = ( tsrs[i] == null ) ? 0.0 : tsrs[i].value64(0);
                if ( value == 1.0) {
                    if ( index >= 0 ) return -1;
                    index = i;
                } else if ( value != 0.0 ) return -1;
            }
            return index;
        }
        return -1;
    }

    private Tsr _apply(Device device, int d, Supplier<Tsr> actor) {
        Tsr out = null;
        if (d >= 0) {
            for (int i = 0; i < _src.size(); i++) {//constants need to be figured out!
                int di = (_src.get(i).dependsOn(d)) ? i : -1;
                if (di >= 0) {
                    if (out == null) out = actor.get();
                    else device.execute(
                            new ExecutionCall( device, new Tsr[]{null, actor.get(), out}, -1, OperationType.instance("+") )
                    );
                }
            }
        } else out = actor.get();
        return out;
    }

    private Tsr[] _src_acti(Tsr[] inputs, int j, int d, int offset) {
        int[] tempShape = null;
        Tsr[] tsrs = new Tsr[ _src.size() + offset ];
        for (int i = offset; i < tsrs.length; i++) {//constants need to be figured out!
            if (!(_src.get(i - offset) instanceof FunctionConstant)) {
                if (d < 0) {
                    tsrs[i] = (j >= 0) ? _src.get(i - offset).call(inputs, j) : _src.get(i - offset).call(inputs);
                } else {
                    tsrs[i] = (j >= 0) ? _src.get(i - offset).derive(inputs, d, j) : _src.get(i - offset).derive(inputs, d);
                }
                tempShape = (tempShape == null) ? tsrs[i].getNDConf().shape() : tempShape;
            }
        }
        for (int i = offset; i < tsrs.length; i++) {
            if (tsrs[i] == null) {
                tsrs[i] =
                        (j < 0)
                                ? new Tsr(tempShape, ((FunctionConstant) _src.get(i - offset)).value())
                                : new Tsr(tempShape, _src.get(i - offset).call(new double[]{}, j));
            }
        }
        return tsrs;
    }

    private Device _device(Tsr[] inputs) {
        Device device = inputs[0].find(Device.class);
        boolean onSameDevice = _shareGuestDevice(inputs);
        boolean doAccel = (!_type.identifier().equals(",") && onSameDevice);
        return (doAccel && device != null) ? device : inputs[0].device();
    }

    private static boolean _shareGuestDevice(Tsr[] tsrs) {
        boolean onSameGuestDevice = true;
        Device device = null;
        for (Tsr tsr : tsrs) {
            device = (tsr.isOutsourced()) ? tsr.find(Device.class) : device;
        }
        if (device != null) {
            for (Tsr tsr : tsrs) {
                onSameGuestDevice = (!tsr.isVirtual() && device == tsr.find(Device.class)) && onSameGuestDevice;
            }
        } else {
            onSameGuestDevice = false;
        }
        if (device != null && tsrs.length == 2 && tsrs[1].size() == 1) {
            onSameGuestDevice = true;
        }
        return onSameGuestDevice;
    }

    //------------------------------------------------------------------------------------------------------------------

    protected double _scalar_activation(double input, boolean derive) {
        switch (_type.identifier()) {
            case "relu":
                return Exec.reLu(input, derive);
            case "sig":
                return Exec.sigmoid(input, derive);
            case "tanh":
                return Exec.tanh(input, derive);
            case "quad":
                return Exec.quadratic(input, derive);
            case "lig":
                return Exec.ligmoid(input, derive);
            case "lin":
                return Exec.linear(input, derive);
            case "gaus":
                return Exec.gaussian(input, derive);
            case "abs":
                return Exec.absolute(input, derive);
            case "sin":
                return Exec.sinus(input, derive);
            case "cos":
                return Exec.cosinus(input, derive);
            default:
                return input;
        }
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    protected double _scalar_activation(double[] input, int j, int d) {
        switch (_type.identifier()) {
            case "sum": return (j < 0) ? Exec.summation(input, d, _src) : Exec.summation(input, j, d, _src);
            case "prod": return (j < 0) ? Exec.PI(input, d, _src) : Exec.PI(input, j, d, _src);
            case "^": return (j < 0) ? Exec.power(input, d, _src) : Exec.power(input, j, d, _src);
            case "/": return (j < 0) ? Exec.division(input, d, _src) : Exec.division(input, j, d, _src);
            case "*": return (j < 0) ? Exec.multiplication(input, d, _src) : Exec.multiplication(input, j, d, _src);
            case "%": return (j < 0) ? Exec.modulo(input, d, _src) : Exec.modulo(input, j, d, _src);
            case "-": return (j < 0) ? Exec.subtraction(input, d, _src) : Exec.subtraction(input, j, d, _src);
            case "+": return (j < 0) ? Exec.addition(input, d, _src) : Exec.addition(input, j, d, _src);
            case "x": return (j < 0) ? Exec.multiplication(input, d, _src) : Exec.multiplication(input, j, d, _src);
            default: return _scalar_activation(input[0], d >= 0);
        }
    }

    public static class Exec
    {
        @Contract(pure = true)
        public static double reLu(double input, boolean derive) {
            double output;
            if ( !derive ) {
                if ( input >= 0 ) output = (input);
                else output = input * 0.01;
                return output;
            } else {
                if ( input >= 0 ) output = 1;
                else output = 0.01;
                return output;
            }
        }

        @Contract(pure = true)
        public static double sigmoid(double input, boolean derive) {
            if ( !derive ) {
                return 1 / (1 + Math.pow(Math.E, -input));
            } else {
                return (Math.pow(Math.E, -input)) / (Math.pow((1 + Math.pow(Math.E, -input)), 2) + 2 * Math.pow(Math.E, -input));
            }
        }

        @Contract(pure = true)
        public static double tanh(double input, boolean derive) {
            if ( !derive ) {
                return input / Math.pow((1 + Math.pow(input, 2)), 0.5);
            } else {
                return (1 - Math.pow((input / Math.pow((1 + Math.pow( input, 2)), 0.5)), 2));
            }
        }

        @Contract(pure = true)
        public static double quadratic(double input, boolean derive) {
            if (!derive) return (input * input);
            else return 2 * input;
        }

        @Contract(pure = true)
        public static double ligmoid(double input, boolean derive) {
            if ( !derive ) return Math.log(1 + Math.pow(Math.E, input));
            else return sigmoid(input, false);
        }

        @Contract(pure = true)
        public static double linear(double input, boolean derive) {
            if ( !derive ) return input;
            else return 1;
        }

        @Contract(pure = true)
        public static double gaussian(double input, boolean derive) {
            if ( !derive ) return Math.pow(Math.E, -Math.pow(input, 2));
            else return -2 * input * Math.pow(Math.E, -Math.pow(input, 2));
        }

        @Contract(pure = true)
        public static double absolute(double input, boolean derive) {
            if ( !derive ) return Math.abs( input );
            else return ( input < 0 ) ? -1 : 1;
        }

        @Contract(pure = true)
        public static double sinus(double input, boolean derive) {
            if ( !derive ) return Math.sin( input );
            else return Math.cos( input );
        }

        @Contract(pure = true)
        public static double cosinus(double input, boolean derive) {
            if ( !derive ) return Math.cos( input );
            else return -Math.sin( input );
        }

        private static double summation(double[] inputs, int j, int d, List<Function> src) {
            if ( d < 0 ) {
                double sum = 0;
                boolean nothingDone = true;
                for ( int i = 0; i < inputs.length; i++ ) {
                    sum += src.get(0).call( inputs, i );
                    nothingDone = false;
                }
                if ( nothingDone ) {
                    return src.get(0).call( inputs );
                }
                return sum;
            } else {
                return src.get(0).derive( inputs, d, j );
            }
        }

        private static double summation(double[] inputs, int d, List<Function> src) {
            if ( d < 0 ) {
                double sum = 0;
                boolean nothingDone = true;
                for (int i = 0; i < inputs.length; i++) {
                    sum += src.get(0).call( inputs, i );
                    nothingDone = false;
                }
                if ( nothingDone ) {
                    return src.get(0).call( inputs );
                }
                return sum;
            } else {
                double sum = 0;
                boolean nothingDone = true;
                for (int i = 0; i < inputs.length; i++) {
                    double r = src.get(0).derive( inputs, d, i );
                    sum += r;
                    nothingDone = false;
                }
                if ( nothingDone ) {
                    return src.get(0).call(inputs);
                }
                return sum;
            }

        }

        @Contract(pure = true)
        private static double PI( double[] inputs, int j, int d, List<Function> src ) {
            if ( d < 0 ) {
                double prod = 1;
                boolean nothingDone = true;
                for (int Ii = 0; Ii < inputs.length; Ii++) {
                    prod *= src.get(0).call( inputs, Ii );
                    nothingDone = false;
                }
                if ( nothingDone ) return src.get(0).call(inputs, j);
                return prod;
            } else {
                double u, ud, v, vd;
                u = src.get(0).call( inputs, 0 );
                ud = src.get(0).derive(inputs, d, 0);
                for (int ji = 1; ji < inputs.length; ji++) {
                    v = src.get(0).call( inputs, ji );
                    vd = src.get(0).derive( inputs, d, ji );
                    ud = u * vd + v * ud;
                    u *= v;
                }
                return ud;
            }
        }

        @Contract(pure = true)
        private static double PI( double[] inputs, int d, List<Function> src ) {
            if (d < 0) {
                double prod = 1;
                boolean nothingDone = true;
                for ( int i = 0; i < inputs.length; i++ ) {
                    prod *= src.get(0).call(inputs, i);
                    nothingDone = false;
                }
                if (nothingDone) return src.get(0).call(inputs);
                return prod;
            } else {
                double u, ud, v, vd;
                u = src.get(0).call(inputs, 0);
                ud = src.get(0).derive(inputs, d, 0);
                for ( int j = 1; j < inputs.length; j++ ) {
                    v = src.get(0).call(inputs, j);
                    vd = src.get(0).derive(inputs, d, j);
                    ud = u * vd + v * ud;
                    u *= v;
                }
                return ud;
            }
        }

        // d/dx(f(x)^g(x))=
        // f(x)^g(x) * d/dx(g(x)) * ln(f(x))
        // + f(x)^(g(x)-1) * g(x) * d/dx(f(x))
        @Contract(pure = true)
        private static double power(double[] inputs, int j, int d, List<Function> src) {
            if (d < 0) {
                double result = src.get(0).call(inputs, j);
                for ( int i = 1; i < src.size(); i++ ) {
                    final double current = src.get(i).call(inputs, j);
                    result = Math.pow(result, current);
                }
                return result;
            } else {
                double out = 0;
                for ( int si = 0; si < src.size(); si++ ) {
                    double b = 1;
                    for ( int i = 1; i < src.size(); i++ ) {
                        b *= src.get(i).call(inputs, j);
                    }
                    if ( si == 0 ) {
                        out += src.get(0).derive(inputs, d, j) * b * Math.pow(src.get(0).call(inputs, j), b - 1);
                    } else {
                        double a = src.get(0).call(inputs, j);
                        out += ( a >= 0 ) ? src.get(si).derive(inputs, d, j) * b * Math.log(a) : 0;
                    }
                }
                return out;
            }
        }

        @Contract(pure = true)
        private static double power(double[] inputs, int d, List<Function> src) {
            if (d < 0) {
                double result = src.get(0).call(inputs);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).call(inputs);
                    result = Math.pow(result, current);
                }
                return result;
            } else {
                double b = 1;
                double bd = 0;
                double a = 0;
                for (int i = 1; i < src.size(); i++) {
                    double dd = 1;
                    a = src.get(i).call(inputs);
                    for (int di = 1; di < src.size(); di++) {
                        if (di != i) dd *= a;
                        else dd *= src.get(di).derive(inputs, d);
                    }
                    bd += dd;
                    b *= a;
                }
                double out = 0;
                a = src.get(0).call(inputs);
                out += src.get(0).derive(inputs, d) * b * Math.pow(a, b - 1);
                out += (a >= 0) ? bd *  Math.pow(a, b) * Math.log(a) : 0;
                return out;
            }
        }

        @Contract(pure = true)
        private static double division(double[] inputs, int j, int d, List<Function> src) {
            if (d < 0) {
                double result = src.get(0).call(inputs, j);
                for (int Vi = 1; Vi < src.size(); Vi++) {
                    final double current = src.get(Vi).call(inputs, j);
                    result /= current;
                }
                return result;
            } else {
                double u, ud, v, vd;
                u = src.get(0).call(inputs, j);
                ud = src.get(0).derive(inputs, d, j);
                for (int i = 0; i < src.size() - 1; i++) {
                    v = src.get(i + 1).call(inputs, j);
                    vd = src.get(i + 1).derive(inputs, d, j);
                    ud = (ud * v - u * vd) / Math.pow(v, 2);
                    u /= v;
                }
                return ud;
            }
        }

        @Contract(pure = true)
        private static double division(double[] inputs, int d, List<Function> src) {
            if (d < 0) {
                double result = src.get(0).call(inputs);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).call(inputs);
                    result /= current;
                }
                return result;
            } else {
                double derivative = 0;
                double tempVar = src.get(0).call(inputs);
                derivative = src.get(0).derive(inputs, d);

                for (int i = 0; i < src.size() - 1; i++) {
                    double u, ud, v, vd;
                    v = src.get(i + 1).call(inputs);
                    vd = src.get(i + 1).derive(inputs, d);
                    u = tempVar;
                    ud = derivative;
                    derivative = (ud * v - u * vd) / Math.pow(v, 2);
                    tempVar /= v;
                }
                return derivative;
            }
        }

        @Contract(pure = true)
        private static double multiplication(double[] inputs, int j, int d, List<Function> src) {
            if (d < 0) {
                double result = src.get(0).call(inputs, j);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).call(inputs, j);
                    result *= current;
                }
                return result;
            } else {
                double u, ud, v, vd;
                u = src.get(0).call(inputs, j);
                ud = src.get(0).derive(inputs, d, j);

                for (int ji = 1; ji < src.size(); ji++) {
                    v = src.get(ji).call(inputs, j);
                    vd = src.get(ji).derive(inputs, d, j);
                    ud = u * vd + v * ud;
                    u *= v;
                }
                return ud;
            }
        }

        @Contract(pure = true)
        private static double multiplication(double[] inputs, int d, List<Function> src) {
            if (d < 0) {
                double result = src.get(0).call(inputs);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).call(inputs);
                    result *= current;
                }
                return result;
            } else {
                double u, ud, v, vd;
                u = src.get(0).call(inputs);
                ud = src.get(0).derive(inputs, d);
                for (int j = 1; j < src.size(); j++) {
                    v = src.get(j).call(inputs);
                    vd = src.get(j).derive(inputs, d);

                    ud = u * vd + v * ud;
                    u *= v; // ...this step can be avoided (TODO optimize)
                }
                return ud;
            }
        }

        @Contract(pure = true)
        private static double idy(double[] inputs, int d, List<Function> src) {
            if (d < 0) {
                inputs[0] = inputs[1];
            } else {
            }
            return 0;
        }

        @Contract(pure = true)
        private static double modulo(double[] inputs, int j, int d, List<Function> src) {
            if (d < 0) {
                double result = src.get(0).call(inputs, j);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).call(inputs, j);
                    result %= current;
                }
                return result;
            } else {
                return src.get(0).derive(inputs, d, j);// j ?
            }
        }

        @Contract(pure = true)
        private static double modulo(double[] inputs, int d, List<Function> src) {
            if (d < 0) {
                double result = src.get(0).call(inputs);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).call(inputs);
                    result %= current;
                }
                return result;
            } else {
                return src.get(0).derive(inputs, d);
            }
        }

        @Contract(pure = true)
        private static double subtraction(double[] inputs, int j, int d, List<Function> src) {
            if (d < 0) {
                double result = src.get(0).call(inputs, j);
                for (int Vi = 1; Vi < src.size(); Vi++) {
                    final double current = src.get(Vi).call(inputs, j);
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
        private static double subtraction(double[] inputs, int d, List<Function> src) {
            if (d < 0) {
                double result = src.get(0).call(inputs);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).call(inputs);
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

        @Contract(pure = true)
        private static double addition(double[] inputs, int j, int d, List<Function> src) {
            if (d < 0) {
                double result = src.get(0).call(inputs, j);
                for (int i = 1; i < src.size(); i++) {
                    final double current = src.get(i).call(inputs, j);
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
        private static double addition(double[] inputs, int d, List<Function> src) {
            if (d < 0) {
                double result = src.get(0).call(inputs);
                for (int Vi = 1; Vi < src.size(); Vi++) {
                    final double current = src.get(Vi).call(inputs);
                    result += current;
                }
                return result;
            } else {
                double derivative = 0;
                for (Function function : src) {
                    derivative += function.derive(inputs, d);
                }
                return derivative;
            }
        }

    }

}
