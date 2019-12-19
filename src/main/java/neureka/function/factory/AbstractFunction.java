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
        if (!_isFlat)//&& d < 0
        {
            /** only flat functions can be executed **/
            if (TYPES.isFunction(_id)) {
                if(d<0){
                    return (FunctionBuilder.build(TYPES.REGISTER[_id] + "(I[" + ((j < 0) ? 0 : j) + "])", true).activate(inputs));
                } else {
                    return _newExec(inputs, d, ((j < 0) ? 0 : j));
                    //return (FunctionBuilder.build(TYPES.REGISTER[_id] + "(I[" + ((j < 0) ? 0 : j) + "])", true).derive(inputs, d));
                }

            } else {
                int[] dx;
                if (TYPES.isFunction(_id)||TYPES.isIndexer(_id)) {
                    /**  SUMMATION, PI,  **/
                    dx = new int[]{d};
                    if(d<0) {
                        Tsr[] tsrs = _source_activation(inputs, dx);
                        return (FunctionBuilder.build(TYPES.REGISTER[_id] + "(I[j])", true).activate(tsrs));
                    } else {
                        return _dxing(dx, (di)->{
                            return _newExec(inputs, d, j);
                            //return (FunctionBuilder.build(TYPES.REGISTER[_id] + "(I[j])", true).derive(tsrs, di));
                        });
                    }
                } else if (TYPES.isOperation(_id)) {
                    /**  '+', '-', 'x', '*', '%', '«', '»', ',', ...  **/
                    String operation = (TYPES.REGISTER[_id].length() > 1) ? TYPES.REGISTER[_id] : "";
                    //dx = new int[_src.size()];
                    //for(int i=0; i<dx.length; i++) dx[i] = d;

                    //String exp = operation;
                    if (j < 0) {
                        if(d<0){
                            Tsr[] tsrs = _source_activation(inputs, j, null, null);
                            for (int i = 0; i < tsrs.length; i++) {
                                operation += "I[" + i + "]" + ((i + 1 < tsrs.length) ? TYPES.REGISTER[_id] : "");
                            }
                            return (FunctionBuilder.build(operation, _doAD).activate(tsrs));
                        } else {
                            return _dxing(_getSrcDx(), (di)->{
                                return _newExec(inputs, d, j);
                                //return (FunctionBuilder.build(exp, _doAD).derive(tsrs, di));
                            });
                        }
                    } else {
                        if(d<0){
                            Tsr[] tsrs = _source_activation(inputs, j, null, null);
                            return (FunctionBuilder.build(operation, _doAD).activate(tsrs, j));
                        } else {
                            return _dxing(_getSrcDx(), (di)->{
                                return _newExec(inputs, d, j);
                                //return (FunctionBuilder.build(exp, _doAD).derive(tsrs, di, j));
                            });
                        }
                    }
                } else {
                    /**  Tensor shape translation:  **/
                    dx = new int[_src.size()];
                    for(int i=0; i<dx.length; i++) dx[i] = d;
                    Tsr[] tsrs = _source_activation(inputs, j, new int[]{1}, null);
                    if (j < 0) {
                        if(d<0){
                            return FunctionBuilder.build(_id, tsrs.length, _doAD).activate(tsrs);
                        } else {
                            return _dxing(dx, (di)->{
                                return _newExec(tsrs, d, j);
                                //return FunctionBuilder.build(_id, tsrs.length, _doAD).derive(tsrs, di);
                            });
                        }
                    } else {
                        if(d<0){
                            return FunctionBuilder.build(_id, tsrs.length, _doAD).activate(tsrs, j);
                        } else {
                            return _dxing(dx, (di)->{
                                return _newExec(tsrs, d, j);
                                //return FunctionBuilder.build(_id, tsrs.length, _doAD).derive(tsrs, di, j);
                            });
                        }
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


    private Tsr _dxing(int[] dx, java.util.function.Function<Integer, Tsr> actor){
        Tsr out = null;
        for(int di : dx){
            if(di>=0){
                if(out==null){
                    out = actor.apply(di);
                } else {
                    Tsr.CPU.execute(new Tsr[]{null, actor.apply(di), out}, TYPES.LOOKUP.get("+"), -1);
                }
            }
        }
        return out;
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
        }
        else
        {
            if (TYPES.REGISTER[_id] == "x") {
                    Tsr tensor1 = _src.get(0).activate(inputs).setIsVirtual(false);
                    Tsr tensor2 = _src.get(1).activate(inputs).setIsVirtual(false);
                    Tsr newTensor = new Tsr(Tsr.fcn.indexing.shpOfCon(tensor1.shape(), tensor2.shape()));
                    Tsr[] array = new Tsr[]{newTensor, tensor1, tensor2};
                    Tsr.CPU.execute(array, _id, d);
                    return array[0];
            } else if (_id == TYPES.LOOKUP.get("<<x") || _id == TYPES.LOOKUP.get("x>>")) {
                if (d < 0) {
                    Tsr[] tsrs = new Tsr[]{
                            _src.get(0).activate(inputs).setIsVirtual(false),
                            _src.get(1).activate(inputs).setIsVirtual(false),
                            _src.get(2).activate(inputs).setIsVirtual(false)
                    };
                    Tsr.CPU.execute(tsrs, _id, 0);

                    if (_id == TYPES.LOOKUP.get("x>>")) {
                        return tsrs[2];
                    } else {
                        return tsrs[0];
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
                Tsr output = _src.get(0).activate(inputs);
                Tsr input =  _src.get(1).activate(inputs);
                Tsr.CPU.execute(new Tsr[]{output, input}, _id, d);
                return output;
            } else {

                double[] inp = new double[inputs.length];
                Tsr output = new Tsr(inputs[0], false);
                //Tsr finalOutput = output;
                double[][] data = new double[inputs.length][];
                for(int i=0; i<data.length; i++) data[i] = inputs[i].value64();

                if(output.is64()){
                    double[] outputValue = output.value64();//.value64();
                    output.foreach((i) -> {
                        for (int ii = 0; ii < inputs.length; ii++) {
                            inp[ii] = data[ii][inputs[ii].i_of_i(i)];//ids[ii]];//i
                        }
                        outputValue[output.i_of_i(i)] = _scalar_activation(inp, j, d);
                    });
                } else {
                    float[] outputValue = output.value32();//.value64();
                    output.foreach((i) -> {
                        for (int ii = 0; ii < inputs.length; ii++) {
                            inp[ii] = data[ii][inputs[ii].i_of_i(i)];//ids[ii]];//i
                        }
                        outputValue[output.i_of_i(i)] = (float)_scalar_activation(inp, j, d);
                    });
                }

                Tsr output0 = null;
                if(true||d>=0){
                    output0 =  _newExec(inputs, d, j);
                    System.out.println(output0+" =?= "+output);
                    //assert output0.toString().equals(output.toString());
                    return  output0;
                }

                return  output;

            }
        }
        //Todo: warning/exception.....
        //return new Tsr(inputs[0], false);//Tsr.fcn.create.newTsr(inputs[0].shape(), inputs[0].translation());
    }

    private Tsr _newExec(Tsr[] inputs, int d, int j){
        Tsr[] tsrs;
        if(Function.TYPES.isIndexer(_id)) tsrs = new Tsr[1 +inputs.length]; else tsrs = new Tsr[1 + _src.size()];
        if(d>=0){
            //Chain-rule (forward ad):
            //inner times out means:
            //first derive source!
            //like so:
            if(Function.TYPES.isIndexer(_id)){
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
            if(Function.TYPES.isIndexer(_id)){
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
            Tsr.CPU.execute(tsrs, _id, d); //(d>=0)
            //At the end:
            //multiply inner times outer:
            tsrs = new Tsr[]{null, inner, tsrs[0]};
            Tsr.CPU.execute(tsrs, TYPES.LOOKUP.get("*"), -1);
            return tsrs[0];
        } else {
            if(Function.TYPES.isIndexer(_id)){
                tsrs = new Tsr[1 +inputs.length];
                if(d<0){
                    for (int i = 1; i < tsrs.length; i++) tsrs[i] = _src.get(0).activate(inputs, i-1);
                } else {
                    for (int i = 1; i < tsrs.length; i++) tsrs[i] = _src.get(0).derive(inputs, d, i-1);
                }
                Tsr.CPU.execute(tsrs, _id, d);
            } else {
                tsrs = new Tsr[1 + _src.size()];
                if(d<0){
                    for (int i = 1; i < tsrs.length; i++) tsrs[i] = (j>=0)?_src.get(i-1).activate(inputs, j):_src.get(i-1).activate(inputs);
                } else {
                    for (int i = 1; i < tsrs.length; i++) tsrs[i] = (j>=0)?_src.get(i-1).derive(inputs, d, j):_src.get(i-1).derive(inputs, d);
                }
                Tsr.CPU.execute(tsrs, _id, d);
            }
        }
        //System.out.println(Function.TYPES.REGISTER[_id]);
        return (tsrs[0]==null)?tsrs[1]:tsrs[0];

    }

    /**
     * **/
    private Tsr[] _source_activation(Tsr[] input, int[] dx) {
        Tsr[] tsrs = new Tsr[input.length];
        int d = -1;
        if(dx!=null&&dx[0]>=0){
            d = (dx[0]>=0)?dx[0]:d;
            dx[0] = (_src.get(0).dependsOn(dx[0]))?0:-1;
        }
        for (int i = 0; i < tsrs.length; i++) {
            if(d<0){
                tsrs[i] = _src.get(0).activate(input, i);
            } else {
                tsrs[i] = _src.get(0).derive(input, d, i);
            }
        }

        return tsrs;
    }

    private int[] _getSrcDx(){
        int[] dx = new int[_src.size()];
        for (int i = 0; i < _src.size(); i++) {//constants need to be figured out!
            if(dx!=null){
                //d = (dx[i]>=0)?dx[i]:d;
                dx[i] = (_src.get(i).dependsOn(dx[i]))?i:-1;
            }
        }
        return dx;
    }

    private Tsr[] _source_activation(Tsr[] input, int j, int[] templateShape, int[] dx) {
        boolean shareDevice = _shareGuestDevice(input);
        Tsr[] tsrs = new Tsr[_src.size()];
        int d = -1;
        for (int i = 0; i < _src.size(); i++) {//constants need to be figured out!
            if(dx!=null){
                d = (dx[i]>=0)?dx[i]:d;
                dx[i] = (_src.get(i).dependsOn(dx[i]))?i:-1;
            }
        }
        for (int i = 0; i < tsrs.length; i++) {//constants need to be figured out!
            if (_src.get(i) instanceof FConstant) {
                tsrs[i] = null;
            } else {
                if(d<0){
                    tsrs[i] = (j < 0) ? _src.get(i).activate(input) : _src.get(i).activate(input, j);
                } else {
                    tsrs[i] = (j < 0) ? _src.get(i).derive(input, d) : _src.get(i).derive(input, d, j);
                }
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
