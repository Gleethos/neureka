package neureka.core.device;
import com.aparapi.Kernel;
import neureka.core.T;

public class TKernel extends Kernel
{
    /**
     * TODO: implement tsr cleanup
     * TODO: implement tsr relu
     * TODO: implement value trim (unalloc_limit)
     * */
    private int biggest_free = 0;
    private float free_ratio = -1;
    private static float unalloc_limit = 1; //  unalloc/alloc should mot be greater than 1
    private static float _alloc_val_sizer = 1.5f;
    private static float alloc_shp_tln_sizer = 1.2f;

    private void calc_biggest_free(){
        int biggest = 0;
        for(int t_i = -1; t_i<tsr_count(); t_i++){
            biggest = (biggest<free_spc(t_i)) ?free_spc(t_i):biggest;
        }
        biggest_free = biggest;
    }

    private void calc_free_ratio(){
        float alloc = 0;
        float free = this.free_spc(-1);
        for(int i=0; i<tsr_count(); i++){
            alloc += this.tsr_grd_end(i)-this.tsr_ptr(i);
            free += this.free_spc(i);
        }
        this.free_ratio =  (free/alloc);

    }
    //---------------------------------------------
    public double[] values(){
        this.get(_values);
        return _values;
    }
    public int[] shapes(){
        this.get(_shapes);//probably obsolete
        return _shapes;
    }
    public int[] translations(){
        this.get(_translations);//probably obsolete ... why? -> they are already present!
        return _translations;
    }
    public int[] pointers(){
        this.get(_pointers);//probably obsolete
        return _pointers;
    }
    public int[] idx(){
        this.get(_idx);//probably obsolete
        return _idx;
    }

    //------------------------------------------
    public double[] value(){this.get(__val); return __val; }
    public int[] shape(){this.get(__shp); return __shp;}
    public int[] translation(){this.get(__tln); return __tln;}
    //------------------------------------------
    private double[] __val;
    private int[] __shp;
    private int[] __tln;
    //-----------------------------
    public @PrivateMemorySpace(8*3) int[] _idx = new int[8*3];
    //-----------------------------
    public int[] _mde = {0};
    /**
     *    TENSORS:
     * */
    public double[] _values;
    public int[] _shapes;
    public int[] _translations;

    public int[] _pointers;// Pointers of tensors (chronologically)
    /**
     *    tsr pointer++:
     *    +0 -> tsr_ptr: for _values
     *    +1 -> tsr_sze: (size) -> negative means: hasGradient()==true
     *       -> full_size: (size) -> 2*tsr_sze if has Gradient()==true
     *
     *    +2 -> shp_ptr: for _shapes
     *    +3 -> shp_sze: (size)
     *
     *    +4 -> tln_ptr: for shape index to value translation
     *    +5 -> tln_sze: (size)
     * */
    public TKernel(){
        this.setExplicit(true);
        _pointers = new int[]{};//0, -initialSize, 0, 0, 0, 0
        _shapes = new int[0];
        _translations = new int[0];
        _values = new double[0];
    }
    //-------------------------------------------------------------------
    public int tsr_count(){
        return _pointers.length/6;
    }
    //-------------------------------------------------------------------
    // Tensors:
    private int tsr_ptr(int t_id){
        return _pointers[t_id*6+0 ];
    }
    private int tsr_sze(int t_id){
        return Math.abs(_pointers[t_id*6+1]);
    }
    private int tsr_end(int t_id){
        return (t_id<0)?0:(tsr_ptr(t_id)+tsr_sze(t_id)+ tsr_grd_sze(t_id));
    }

    private void set_tsr_ptr(int t_id, int ptr){
        _pointers[t_id*6+0 ] = ptr;
    }
    private void set_tsr_sze(int t_id, int sze){
        _pointers[t_id*6+1]=sze;}

    // Gradients:
    private int tsr_grd_ptr(int t_id){
        return tsr_ptr(t_id)+tsr_sze(t_id);
    }
    private int tsr_grd_sze(int t_id){return tsr_sze(t_id)*((hasGradient(t_id))?1:0);}
    private int tsr_grd_end(int t_id){return tsr_grd_ptr(t_id)+tsr_sze(t_id);}

    // Shapes:
    private int shp_ptr(int t_id){
        return _pointers[t_id*6+2 ];
    }
    private int shp_sze(int t_id){
        return _pointers[t_id*6+3];
    }
    private int shp_end(int t_id){return shp_ptr(t_id)+shp_sze(t_id);}

    private void set_shp_ptr(int t_id, int ptr){
        _pointers[t_id*6+2 ] = ptr;
    }
    private void set_shp_sze(int t_id, int sze){
        _pointers[t_id*6+3]=sze;}

    // Translations:
    private int tln_ptr(int t_id){
        return _pointers[t_id*6+4 ];
    }
    private int tln_sze(int t_id){
        return _pointers[t_id*6+5];
    }
    private int tln_end(int t_id){return tln_ptr(t_id)+tln_sze(t_id);}

    private void set_tln_ptr(int t_id, int ptr){
        _pointers[t_id*6+4 ] = ptr;
    }
    private void set_tln_sze(int t_id, int sze){
        _pointers[t_id*6+5]=sze;}
    //-------------------------------------------------------------------
    private int free_spc(int t_id){
        return  ((t_id+1)*6<_pointers.length)
                ?(tsr_ptr(t_id+1)-tsr_end(t_id))//=> getting space between t_id and next element
                :_values.length-tsr_end(t_id);//=> t_id is last element
    }
    //-------------------------------------------------------------------
    private void setNull(int t_id){
        _pointers[t_id*6+1]=0;
    }
    private boolean ptrIsNull(int t_id){
        return (t_id<0)?false:(_pointers[t_id*6+1]==0);
    }
    private boolean hasGradient(int t_id){
        return (_pointers[t_id*6+1]<0);
    }
    //-------------------------------------------------------------------
    /**
     *    pointer modification (allocation and freeing)
     *    ------------------------------------------------------
     * */
    public int freePtrOf(int t_id, int[][] regis){
        return mod_ptrs(t_id, true, regis);
    }

    public int allocPtrFor(T tensor, int[][] regis){
        int size = tensor.value().length;
        int[] shape = tensor.shape();
        int[] translation = tensor.translation();
        __shp = shape;
        __tln = translation;
        this.put(__shp);
        this.put(__tln);
        int biggestChunck = 0;
        for(int t_i = -1; t_i<tsr_count(); t_i++){
            biggestChunck =
                (biggestChunck<free_spc(t_i))
                    ?free_spc(t_i)
                    :biggestChunck;
        }
        if(biggestChunck<size){
            int newSpace = (int)(_values.length* _alloc_val_sizer);
            newSpace = (newSpace>size)?newSpace:size;
            this.get(_values);//TODO: Make a flag so that this is avoided!
            double[] newValues = new double[_values.length+newSpace];
            for(int i = 0; i< _values.length; i++){
                newValues[i] = _values[i];
            }
            _values = newValues;
            if(_pointers.length==0){
                _pointers = new int[]{0, 0, 0, 0, 0, 0};
            }
            this.put(_values);
        }
        for(int t_i = -1; t_i<tsr_count(); t_i++){//  t_i=-1  :
            if(free_spc(t_i)>=size){
                int ptr = 0;
                if(ptrIsNull(t_i)==false){
                    ptr = mod_ptrs(t_i, false, regis);
                    t_i++;
                }else{
                    regis[0][0] = 0;
                }
                set_tsr_ptr(t_i, tsr_end(((t_i>0)?t_i-1:0)));
                set_tsr_sze(t_i, size*((tensor.rqsGradient())?-1:1));
                set_shp_ptr(t_i, alloc_shp(shape));
                set_shp_sze(t_i, shape.length);
                set_tln_ptr(t_i, alloc_tln(translation));
                set_tln_sze(t_i, translation.length);
                this.put(_pointers);
                return ptr;
            }
        }
        this.put(_pointers);
        return 0;//return pointer f _alloc_val_sizer
    }

    private int mod_ptrs(int t_id, boolean rmv, int[][] regis){
        int[] mapper = new int[regis[0].length];//tsr_count()
        int rgr_ptr = 0;
        for(int i=0; i<mapper.length; i++){
            if(regis[0][i]>=0){//=> REGISTER contains t_id's or null pointer (-1)
                mapper[regis[0][i]] = i;
                //=> mapper points from pointer entries to REGISTER entries
            }
        }
        int[] newPointers;
        if(rmv){//Removing pointer entry and setting REGISTER entry to null (-1)
            regis[0][mapper[t_id]]=-1;
            newPointers = new int[_pointers.length-6];
            for(int i=0; i<t_id*6; i++){
                newPointers[i] = _pointers[i];
            }
            for(int i = t_id*6+6; i<_pointers.length; i++) {
                newPointers[i-6] = _pointers[i];
                if(i%6==0){
                    regis[0][mapper[(i/6)]]--;
                }
            }
        }else{
            boolean registered = false;
            for(int i=0; i<regis[0].length; i++){
                if(regis[0][i]<0){//Null pointer found in REGISTER!
                    rgr_ptr = i;//
                    regis[0][i]=t_id+1;//mapper[t_id+1] = i;
                    registered = true;
                    i = regis[0].length;
                }
            }
            if(registered==false){
                int[] newRegister = new int[regis[0].length+6];
                for(int i=0; i<newRegister.length; i++){
                    newRegister[i] = (regis[0].length>i)?regis[0][i]:-1;
                }
                rgr_ptr = regis[0].length;
                newRegister[regis[0].length]=t_id+1;
                regis[0] = newRegister;

            }
            newPointers = new int[_pointers.length+6];
            for(int i=0; i<t_id*6+6; i++){
                newPointers[i] = _pointers[i];
            }
            for(int i=t_id*6+6; i<newPointers.length-6; i++){
                newPointers[i+6] = _pointers[i];
                if(i%6==0){
                    regis[0][mapper[(i/6)]]++;
                }
            }
        }
        _pointers = newPointers;
        this.put(_pointers);
        return rgr_ptr;//mapper[t_id];
    }
    /**
     *    returns pointer to _shapes/_translations elements.
     *    moves _shapes/_translations to device if required data is not present.
     *
     * */
    private int alloc_tln(int[] translation){
        int zeros_ptr = 0;
        for(int i = 0; i<_translations.length; i++){
            if(_translations[i]!=0){
                zeros_ptr = i+1;
            }
            boolean matches = true;
            for(int ii=0; ii<translation.length; ii++){
                if((i+ii>=_translations.length)||_translations[i+ii]!=translation[ii]&&_translations[i+ii]!=0){
                    matches=false;
                }
            }
            if(matches){
                for(int ii=0; ii<translation.length; ii++){
                    _translations[i+ii]=translation[ii];
                }
                this.put(_translations);
                return i;
            }
        }
        int[] newTranslations
                = new int[
                _translations.length+
                        (
                                translation.length>((int)(_translations.length* alloc_shp_tln_sizer))
                                        ?translation.length
                                        :((int)(_translations.length* alloc_shp_tln_sizer))
                        )
                ];
        for(int i=0; i<zeros_ptr; i++){
            newTranslations[i] = _translations[i];
        }
        for(int i=zeros_ptr; i<zeros_ptr+translation.length; i++){
            newTranslations[i] =
                    (i<zeros_ptr+translation.length)
                    ?translation[i-zeros_ptr]
                    :0;
        }
        int ptr = zeros_ptr;
        _translations = newTranslations;
        this.put(_translations);
        return ptr;
    }
    private int alloc_shp(int[] shape){
        int zeros_ptr = 0;
        for(int i = 0; i<_shapes.length; i++){
            if(_shapes[i]!=0){
                zeros_ptr = i+1;
            }
            boolean matches = true;
            for(int ii=0; ii<shape.length; ii++){
                if((i+ii>=_shapes.length)||_shapes[i+ii]!=shape[ii]&&_shapes[i+ii]!=0){
                    matches=false;
                }
            }
            if(matches){
                for(int ii=0; ii<shape.length; ii++){
                    _shapes[i+ii]=shape[ii];
                }
                this.put(_shapes);
                return i;
            }
        }
        int[] newShapes =
            new int[
            _shapes.length+
                (
                    shape.length>((int)(_shapes.length* alloc_shp_tln_sizer))
                        ?shape.length
                        :((int)(_shapes.length* alloc_shp_tln_sizer))
                )
            ];
        for(int i=0; i<zeros_ptr; i++){
            newShapes[i] = _shapes[i];
        }
        for(int i=zeros_ptr; i<zeros_ptr+shape.length; i++){
            newShapes[i] =
                (i<zeros_ptr+shape.length)
                    ?shape[i-zeros_ptr]
                    :0;
        }
        int ptr = zeros_ptr;
        _shapes = newShapes;
        this.put(_shapes);
        return ptr;
    }

    /**
     *    Pre-Execution functions (mode setter)
     *    return global size for range creation!
     *    ------------------------------------------------------
     *    ======================================================
    * */
    public int executionSizeOf_fetchTsr(int t_id, boolean grd){
        __val = new double[tsr_sze(t_id)];
        __shp = new int[shp_sze(t_id)];
        __tln = new int[tln_sze(t_id)];
        _mde = new int[]{(grd)?-4:-3, t_id};
        this.put(__val).put(__shp).put(__tln).put(_mde);
        int g_sze = tsr_sze(t_id)+shp_sze(t_id)+tln_sze(t_id);
        //System.out.println("fetch: "+g_sze);
        return g_sze;
    }

    public int executionSizeOf_storeTsr(int t_id, double[] value, boolean grd){
        _mde = new int[]{(grd)?-2:-1, t_id};// 1. define if stored as grd or not; 2. specify tsr id;
        __val = value;
        this.put(_mde).put(__val);
        int g_sze = tsr_sze(t_id)+shp_sze(t_id)+tln_sze(t_id);
        //System.out.println(g_sze);
        return g_sze;
    }

    /**
     * Mode format:
     * 0: f_id
     * 1-(n-1): t_id's
     * n: d (-1 if not derivative)
     * */
    public int executionSizeOf_calc(int[] mode){// Mode contains f_id, drain id and source id's !
        if(_mde ==null||_mde.length<3||_mde.length!=mode.length){
            _mde = mode;
            this.put(_mde);//up
        }
        for(int i = 0; i< _idx.length; i++){
            _idx[i] = 0;
        }
        this.put(_idx);
        return tsr_sze(mode[1]);
    }

    /**
     * Mode format:
     * 0: f_id
     * 1-(n-1): t_id's
     * n: d (-1 if not derivative)
     * */
    public int executionSizeOf_calc(int[] mode, double value){// Mode contains f_id, value is applied to all!
        if(_mde ==null||_mde.length<3||_mde.length!=mode.length){
            _mde = mode;
            this.put(_mde);//up
        }
        for(int i = 0; i< _idx.length; i++){
            _idx[i] = 0;
        }
        this.put(_idx);
        __val = new double[]{value};
        this.put(__val);
        return tsr_sze(mode[1]);
    }

    /**
     *    KERNEL RUN:
     *    ==========
     * */

    @Override
    public void run() {
        run(this.getGlobalId());
    }

    public void run(int gid, int[] m){
        _mde = m;
        run(gid);
    }

    private void run(int gid){
        for(int i = 0; i< _idx.length; i++){
            _idx[i] = 0;
        }
        if(_mde[0]==-5){//cleanup //TODO implement!
        }
        if(_mde[0]==-4){//executionSizeOf_fetchTsr grad f tensor
            run_fetch(gid, true);
        }
        if(_mde[0]==-3){//executionSizeOf_fetchTsr tensor
            run_fetch(gid, false);
            //__val[gid] = gid;
        }
        if(_mde[0]==-2){//executionSizeOf_storeTsr grad f tensor
            run_store(gid, true);
        }
        if(_mde[0]==-1){//executionSizeOf_storeTsr tensor
            run_store(gid, false);
        }
        if(_mde[0]==0){//Relu
            run_relu(gid,  _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==1){//Sigmoid
            run_sig(gid,  _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==2){//Tanh
            run_tnh(gid,  _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==3){//Quadratic
            run_qdr(gid,  _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==4){//Ligmoid
            run_lig(gid,  _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==5){//Linear
            run_lin(gid,  _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==6){//Gaussian
            run_gus(gid,  _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==7){//Absolut
            run_abs(gid,  _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==8){//Sinus
            run_sin(gid,  _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==9){//Cosinus
            run_cos(gid,  _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==10){//Sum
            run_sum(gid,  _mde[1], (_mde.length<=3)?-1: _mde[_mde.length-1]);
        }
        if(_mde[0]==11){//Product
            run_pi(gid,  _mde[1], (_mde.length<=3)?-1: _mde[_mde.length-1]);
        }
        if(_mde[0]==12){//  ^
            run_pow(gid,  _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==13){//  /
            if(_mde.length>2){
                run_div(gid,  _mde[1], _mde[2], _mde[3]);
            } else {
                run_broadcast_div(gid, _mde[1], __val[0]);
            }
        }
        if(_mde[0]==14){//  *
            if(_mde.length>2) {
                run_mul(gid, _mde[1], _mde[2], _mde[3]);
            }else{
                run_broadcast_mul(gid, _mde[1], __val[0]);
            }
        }
        if(_mde[0]==15){//  %
            if(_mde.length>2) {
                run_mod(gid, _mde[1], _mde[2], _mde[3]);
            } else{
                run_broadcast_mod(gid, _mde[1], __val[0]);
            }
        }
        if(_mde[0]==16){//  -
            if(_mde.length>2) {
                run_sub(gid, _mde[1], _mde[2], _mde[3]);
            }else{
                run_broadcast_sub(gid, _mde[1], __val[0]);
            }
        }
        if(_mde[0]==17){//  +
            if(_mde.length>2) {
                run_add(gid, _mde[1], _mde[2], _mde[3]);
            } else {
                run_broadcast_add(gid, _mde[1], __val[0]);
            }
        }
        if(_mde[0]==18){// x  tsr_conv
            if(_mde[_mde.length-1]<0){
                run_conv(gid, _mde[1], _mde[2], _mde[3]);
            } else {
                run_conv_inv(gid, _mde[1], _mde[2], _mde[3], (_mde[_mde.length-1]==0));
            }
        }

    }
    /*
	     0:  ReLu;
		 1:  Sigmoid;
		 2:  Tanh;
		 3:  Quadratic;
		 4:  Ligmoid;
		 5:  Linear;
		 6:  Gaussian;
		 7:  abs;
		 8:  sin;
		 9:  cos;
		 10: sum;
		 11: prod;
		 12: ^;
		 13: /;
		 14: *;
		 15: %;
		 16: -;
		 17: +;
		 18: tsr mul;
	 */
    private void run_cleanup(int gid){
        //TODO: implement
        //TODO write test cases!
    }

    private void run_fetch(int gid, boolean grd){
        if(gid<tsr_sze(_mde[1])){
            __val[gid]=_values[((!grd)?tsr_ptr(_mde[1]):tsr_grd_ptr(_mde[1]))+gid];
        }else{
            if(gid<(tsr_sze(_mde[1])+shp_sze(_mde[1]))){
                gid-=tsr_sze(_mde[1]);
                __shp[gid]=_shapes[shp_ptr(_mde[1])+gid];
            }else{
                if(gid<(tsr_sze(_mde[1])+shp_sze(_mde[1])+tln_sze(_mde[1]))){
                    gid-=(tsr_sze(_mde[1])+shp_sze(_mde[1]));
                    __tln[gid]=_translations[tln_ptr(_mde[1])+gid];
                }
            }
        }
    }

    private void run_store(int gid, boolean grd){
        if(gid<tsr_sze(_mde[1])){
            _values[((!grd)?tsr_ptr(_mde[1]):tsr_grd_ptr(_mde[1]))+gid]=__val[gid];
        }else{
            if(gid<(tsr_sze(_mde[1])+shp_sze(_mde[1]))){
                gid-=tsr_sze(_mde[1]);
                _shapes[shp_ptr(_mde[1])+gid]=__shp[gid];
            }else{
                if(gid<(tsr_sze(_mde[1])+shp_sze(_mde[1])+tln_sze(_mde[1]))){
                    gid-=(tsr_sze(_mde[1])+shp_sze(_mde[1]));
                    _translations[tln_ptr(_mde[1])+gid]=__tln[gid];
                }
            }
        }
    }

    private void run_relu(int gid, int drn_id, int src_id, int d){
        if (d<0) {
            if (_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)] >= 0) {
                _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                        (_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]);
            } else {
                _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                        (_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]) * 0.01;
            }
        } else {
            if (_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)] >= 0) {
                _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = 0.01;
            } else {
                _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = 0.01;
            }
        }
    }

    private void run_sig(int gid, int drn_id, int src_id, int d){
        if(d<0){
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    1 / (1 + Math.pow(Math.E, (-_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)])));

        }else{
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                (
                    Math.pow(
                        Math.E,
                        -_values[tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)]
                    )
                ) / (Math.pow(
                        (1 + Math.pow(
                                Math.E,
                                -_values[tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)]
                        )
                    ), 2)
                        + 2 * Math.pow(
                                Math.E, -_values[tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)]));
        }
    }

    private void run_tnh(int gid, int drn_id, int src_id, int d){
        if(d<0){
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    _values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                            / Math.pow(
                                    (1 + Math.pow(
                                            _values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                                            , 2)
                                    ), 0.5);

        }else{
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                (1 - Math.pow(
                    (_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                        /
                        Math.pow(
                            (1 + Math.pow(
                                _values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                                , 2)
                            ), 0.5
                        )
                    ), 2)
                );

        }
    }
    private void run_qdr(int gid, int drn_id, int src_id, int d){
        if(d<0){
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.pow(_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)],2);
        }else{
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    _values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]*2;
        }
    }
    private void run_lig(int gid, int drn_id, int src_id, int d){
        if(d<0){
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = (
                _values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                + (
                    Math.log(
                        Math.pow(
                            Math.E,
                            -_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                        ) + 1
                    ) / Math.log(Math.E)
                )
            );
        }else{
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                1 /
                    (1 + Math.pow(
                            Math.E,
                            _values[tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)]
                    )
                );
        }
    }

    private void run_lin(int gid, int drn_id, int src_id, int d){
        if(d<0){
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    _values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)];
        }else{
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    _values[tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)];
        }
    }

    private void run_gus(int gid, int drn_id, int src_id, int d){
        if(d<0){
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    Math.pow(Math.E, -Math.pow(_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)], 2));
        }else{
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    -2 * (_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)])
                            * Math.pow(Math.E, -Math.pow(_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)], 2));
        }
    }

    private void run_abs(int gid, int drn_id, int src_id, int d){
        _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.abs(_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]);
    }
    private void run_sin(int gid, int drn_id, int src_id, int d){
        _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.sin(_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]);
    }
    private void run_cos(int gid, int drn_id, int src_id, int d){
        _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.cos(_values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]);
    }

    private void run_sum(int gid, int drn_id, int d){
        if(d<0){
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = 0;
            for(int i = 2; i<(_mde.length-1); i++){
                _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] +=
                        _values[tsr_ptr(_mde[i])+__i_of(gid, _mde[i], 1)];
            }
        }else{
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    _values[tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)];
        }
    }

    private void run_pi(int gid, int drn_id, int d){
        if(d<0){
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = 0;
            for(int i = 2; i<(_mde.length-1); i++){
                _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] *=
                        _values[tsr_ptr(_mde[i])+__i_of(gid, _mde[i], 1)];
            }
        }else{
            //TODO: implement
            _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    _values[tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)];//........
        }
    }

    private void run_pow(int gid, int drn_id, int src1_id, int src2_id){
        _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.pow(
                        _values[tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)],
                        _values[tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)]
                        );
    }
    private void run_broadcast_pow(int gid, int drn_id, double value){
        _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.pow(_values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)], value);
    }
    private void run_div(int gid, int drn_id, int src1_id, int src2_id){
        _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                _values[tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)]
                        /
                _values[tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)];
    }
    private void run_broadcast_div(int gid, int drn_id, double value){
        _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)]/value;
    }
    private void run_mul(int gid, int drn_id, int src1_id, int src2_id){
        _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                _values[tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)]
                        *
                _values[tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)];
    }
    private void run_broadcast_mul(int gid, int drn_id, double value){
       // _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)]*value;
    }
    private void run_mod(int gid, int drn_id, int src1_id, int src2_id){
        _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                ((int)_values[tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)])
                        %
                ((int)_values[tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)]);
    }
    private void run_broadcast_mod(int gid, int drn_id, double value){
       // _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
       //         (int)(_values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)])%(int)value;
    }
    private void run_sub(int gid, int drn_id, int src1_id, int src2_id){
        _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                _values[tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)]
                        -
                _values[tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)];
    }
    private void run_broadcast_sub(int gid, int drn_id, double value){
      //  _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)]
       //         = _values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)]-value;
    }
    private void run_add(int gid, int drn_id, int src1_id, int src2_id){
        int i1 = tsr_ptr(drn_id)+__i_of(gid, drn_id, 0);
        int i2 = tsr_ptr(src1_id)+__i_of(gid, src1_id, 1);
        int i3 = tsr_ptr(src2_id)+__i_of(gid, src2_id, 2);
        _values[i1] = _values[i2] + _values[i3];
    }
    private void run_broadcast_add(int gid, int drn_id, double value){
      //  int i1 = tsr_ptr(drn_id)+__i_of(gid, drn_id, 0);
      //  _values[i1] = _values[i1]+value;
    }
    //==================================================================================================================
    private void run_conv_inv(int gid, int drn_id, int src1_id, int src2_id, boolean first){

        drn_id = drn_id^src2_id; src2_id = drn_id^src2_id; drn_id = drn_id^src2_id;
        if(first){
            src1_id = src1_id^drn_id; drn_id = src1_id^drn_id; src1_id = src1_id^drn_id;
        }
        // SETUP:
        int p_data_src1 = tsr_ptr(src1_id);
        int p_data_src2 = tsr_ptr(src2_id);
        int p_data_drn = tsr_ptr(drn_id);

        int p_shp_src1 = shp_ptr(src1_id);
        int p_shp_src2 = shp_ptr(src2_id);
        int p_shp_drn  = shp_ptr(drn_id);

        int p_tln_src1 = tln_ptr(src1_id);
        int p_tln_src2 = tln_ptr(src2_id);
        int p_tln_drn  = tln_ptr(drn_id);

        int rank = shp_sze(drn_id);
        int p_idx_src1 = 0*rank;
        int p_idx_src2 = 1*rank;
        int p_idx_drn  = 2*rank;

        int src1End = p_shp_src1 + rank;
        int src2End = p_shp_src2 + rank;

        //increment on drain:
        for(int i=0; i<gid; i++){//drnSze-1
            __increment_idx(p_shp_drn, p_idx_drn, rank);
        }
        //increment src accordingly:
        int ri = 0;
        while (ri < rank) {
            if (_idx[(p_idx_src2+ri)] == _shapes[(p_shp_src2+ri)]) {//_idx[(p_idx_src1+ri)] == _shapes[(p_shp_src1+ri)] ||
                _idx[(p_idx_src2 + ri)] = 0;
                _idx[(p_idx_src1 + ri)] = _idx[(p_idx_drn + ri)];
            } else {
                if (_shapes[(p_shp_drn+ri)] > _shapes[(p_shp_src1+ri)]) {//TODO:THIS IS ADDED
                    _idx[(p_idx_src1+ri)] = (_idx[(p_idx_drn+ri)] - _idx[(p_idx_src2+ri)]);
                } else {
                    _idx[(p_idx_src1+ri)] = (_idx[(p_idx_drn+ri)] + _idx[(p_idx_src2+ri)]);
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
            ri = (ri==rank)?0:ri;
            if (incrementing == false) {
                boolean isMatch = true;
                for(int i=0; i<rank; i++){
                    if(!(_idx[(p_idx_src1+i)] < _shapes[(p_shp_src1+i)] && _idx[(p_idx_src1+i)]>=0)){
                        isMatch = false;
                    }
                }
                if(isMatch){
                    int i1 = __i_of_idx_on_tln(p_tln_src1, p_idx_src1, rank); //(int p_tln, int[] tln, int[] idx, int rank)
                    int i2 = __i_of_idx_on_tln(p_tln_src2, p_idx_src2, rank);
                    value += _values[(p_data_src1 + i1)] * _values[(p_data_src2 + i2)];
                    //1*-2 +2*3 -3*6 +2*3, 1*3 +2*6 -3*3 +2*-1,
                    //1*0  +2*2 -3*4 +2*2  +  4*-2 -2*3 -1*6 +5*3, 1*2 +2*4 -3*2 +2*1  +  4*3 -2*6 -1*3 +5*-1,
                    //4*0  -2*2 -1*4 +5*2, 4*2 -2*4 -1*2 +5*1
                }
                incrementing = true;
                ri=0;
            } else {//incrementing:
                if (_idx[(p_idx_src2+ri)] < _shapes[(p_shp_src2+ri)]) {//_idx[(p_idx_src1+ri)] < _shapes[(p_shp_src1+ri)] &&
                    //_idx[(p_idx_src1+ri)]++;
                    _idx[(p_idx_src2+ri)]++;

                    if (_idx[(p_idx_src2+ri)] == _shapes[(p_shp_src2+ri)]) {//_idx[(p_idx_src1+ri)] == _shapes[(p_shp_src1+ri)] ||
                        if (((p_shp_src2+ri) == (src2End - 1))) {//(p_shp_src1+ri) == (src1End - 1) ||
                            running = false;
                        }
                        _idx[(p_idx_src2+ri)] = 0;
                        _idx[(p_idx_src1+ri)] = _idx[(p_idx_drn+ri)];
                        ri++;
                    } else {
                        if (_shapes[(p_shp_drn+ri)] > _shapes[(p_shp_src1+ri)]) {//TODO:THIS IS ADDED
                            _idx[(p_idx_src1+ri)] = (_idx[(p_idx_drn+ri)] - _idx[(p_idx_src2+ri)]);
                        } else {
                            _idx[(p_idx_src1+ri)] = (_idx[(p_idx_drn+ri)] + _idx[(p_idx_src2+ri)]);
                        }
                        incrementing = false;
                        ri=0;
                    }
                } else {
                    ri++;
                }
            }
        }//-8, 4, -9, -2, 2, 3
        //set value in drn:
        int di = __i_of_idx_on_tln(p_tln_drn, p_idx_drn, rank);
        _values[(p_data_drn + di)] = value;
    }//=================================================================================================================

    private void run_conv(int gid, int drn_id, int src1_id, int src2_id){
        // SETUP:
        int p_data_src1 = tsr_ptr(src1_id);
        int p_data_src2 = tsr_ptr(src2_id);
        int p_data_drn = tsr_ptr(drn_id);

        int p_shp_src1 = shp_ptr(src1_id);
        int p_shp_src2 = shp_ptr(src2_id);
        int p_shp_drn  = shp_ptr(drn_id);

        int p_tln_src1 = tln_ptr(src1_id);
        int p_tln_src2 = tln_ptr(src2_id);
        int p_tln_drn  = tln_ptr(drn_id);

        int rank = shp_sze(drn_id);
        int p_idx_src1 = 0*rank;
        int p_idx_src2 = 1*rank;
        int p_idx_drn  = 2*rank;

        int src1End = p_shp_src1 + rank;
        int src2End = p_shp_src2 + rank;

        //increment on drain:
        for(int i=0; i<gid; i++){//drnSze-1
            __increment_idx(p_shp_drn, p_idx_drn, rank);
        }
        //increment src accordingly:
        int ri = 0;
        while (ri < rank) {
            if (_shapes[(p_shp_src1+ri)] == _shapes[(p_shp_src2+ri)]) {//setting 0
                _idx[(p_idx_src1+ri)] = _idx[(p_idx_drn+ri)];
                _idx[(p_idx_src2+ri)] = _idx[(p_idx_drn+ri)];
            } else if (_shapes[(p_shp_src1+ri)] > _shapes[(p_shp_src2+ri)]) {//setting src1 idx to id idx
                _idx[(p_idx_src1+ri)] = _idx[(p_idx_drn+ri)];
                _idx[(p_idx_src2+ri)] = 0;
            } else if (_shapes[p_shp_src1+ri] < _shapes[(p_shp_src2+ri)]) {//setting src2 idx to id idx
                _idx[(p_idx_src1+ri)] = 0;
                _idx[(p_idx_src2+ri)] = _idx[(p_idx_drn+ri)];
            }
            ri++;
        }
        //----------
        // multiplication:
        double value = 0;
        boolean running = true;
        boolean incrementing = false;
        while (running) {
            ri = (ri==rank)?0:ri;
            if (incrementing == false) {
                int i1 = __i_of_idx_on_tln(p_tln_src1, p_idx_src1, rank); //(int p_tln, int[] tln, int[] idx, int rank)
                int i2 = __i_of_idx_on_tln(p_tln_src2, p_idx_src2, rank);
                value +=
                     _values[(p_data_src1 + i1)]
                         *
                     _values[(p_data_src2 + i2)];
                incrementing = true;
                ri=0;
            } else {//incrementing:
                if (_idx[(p_idx_src1+ri)] < _shapes[(p_shp_src1+ri)] && _idx[(p_idx_src2+ri)] < _shapes[(p_shp_src2+ri)]) {
                    _idx[(p_idx_src1+ri)]++;
                    _idx[(p_idx_src2+ri)]++;
                    if (_idx[(p_idx_src1+ri)] == _shapes[(p_shp_src1+ri)] || _idx[(p_idx_src2+ri)] == _shapes[(p_shp_src2+ri)]) {
                        if (((p_shp_src1+ri) == (src1End - 1) || (p_shp_src2+ri) == (src2End - 1))) {
                            running = false;
                        }
                        if (_shapes[(p_shp_src1+ri)] == _shapes[(p_shp_src2+ri)]) {//setting 0
                            _idx[(p_idx_src1+ri)] = _idx[(p_idx_drn+ri)];//mtch[mi];
                            _idx[(p_idx_src2+ri)] = _idx[(p_idx_drn+ri)];//mtch[mi];
                        } else if (_shapes[(p_shp_src1+ri)] > _shapes[(p_shp_src2+ri)]) {//setting hdr1 idx to id idx
                            _idx[(p_idx_src1+ri)] = _idx[(p_idx_drn+ri)];//mtch[mi];
                            _idx[(p_idx_src2+ri)] = 0;
                        } else if (_shapes[(p_shp_src1+ri)] < _shapes[(p_shp_src2+ri)]) {//setting hdr2 idx to id idx
                            _idx[(p_idx_src1+ri)] = 0;
                            _idx[(p_idx_src2+ri)] = _idx[(p_idx_drn+ri)];//mtch[mi];
                        }
                        ri++;
                    } else {
                        incrementing = false;
                        ri=0;
                    }
                } else {
                    ri++;
                }
            }
        }
        //set value in drn:
        int di = __i_of_idx_on_tln(p_tln_drn, p_idx_drn, rank);
        _values[(p_data_drn + di)] = value;
    }

    //Helper methods for tsr conv:
    private int __increment_At(int ri, int idx_ptr, int shp_ptr) {
        if (_idx[idx_ptr+ri] < (_shapes[shp_ptr+ri])) {//fixed
            _idx[idx_ptr+ri]++;
            if (_idx[idx_ptr+ri] == (_shapes[shp_ptr+ri])) {
                _idx[idx_ptr+ri] = 0;
                ri++;
            } else {
                ri = -1;
            }
        } else {
            ri++;
        }
        return ri;
    }
    private void __increment_idx(int shp_ptr, int idx_ptr, int rank) {
        int ri = 0;
        while (ri >= 0 && ri < rank) {//end
            ri = __increment_At(ri, idx_ptr, shp_ptr);
        }
    }
    private int __i_of_idx_on_tln(int p_tln, int idx_ptr, int rank) {
        int i = 0;
        for (int ii = 0; ii < rank; ii++) {
            i += _translations[p_tln+ii] * _idx[idx_ptr+ii];
        }
        return i;
    }

    private int __i_of(int gid, int t_id, int num){
        int p_shp  = shp_ptr(t_id);
        int p_tln  = tln_ptr(t_id);
        int rank     = shp_sze(t_id);
        int p_idx  = rank*num;
        for(int i=0; i<gid; i++){
            __increment_idx(p_shp, p_idx, rank);
        }
        return __i_of_idx_on_tln(p_tln, p_idx, rank);
    }
    
}
