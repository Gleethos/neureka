package neureka.core.device;
import neureka.core.T;

public class TensorKernel extends com.aparapi.Kernel
{
    /**
     * TODO: implement tsr cleanup
     * TODO: implement _value trim (_unalloc_limit)
     * */
    private int _biggest_free = 0;
    private float _free_ratio = -1;
    private static float _unalloc_limit = 1; //  unalloc/alloc should mot be greater than 1
    private static float _alloc_val_sizer = 1.5f;
    private static float alloc_shp_tln_sizer = 1.2f;

    private void _calc_biggest_free(){
        int biggest = 0;
        for(int t_i = -1; t_i< _tsr_count(); t_i++){
            biggest = (biggest <_free_spc(t_i)) ? _free_spc(t_i):biggest;
        }
        _biggest_free = biggest;
    }

    private void _calc_free_ratio(){
        float alloc = 0;
        float free = this._free_spc(-1);
        for(int i = 0; i< _tsr_count(); i++){
            alloc += this._tsr_grd_end(i)-this._tsr_ptr(i);
            free += this._free_spc(i);
        }
        this._free_ratio =  (free/alloc);
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
    public @PrivateMemorySpace(8*2) int[] _idx = new int[8*2];
    //-----------------------------
    public int[] _mde = {0};

    /**    TENSORS:   * */
    public double[] _values;
    public int[] _shapes;
    public int[] _translations;

    /**   POINTERS:   **/
    public int[] _pointers;// Pointers of tensors (chronologically)
    /**
     *    tsr pointer++:
     *    +0 -> _tsr_ptr: for _values
     *    +1 -> _tsr_sze: (size) -> negative means: _has_gradient()==true
     *       -> full_size: (size) -> 2*_tsr_sze if has Gradient()==true
     *
     *    +2 -> _shp_ptr: for _shapes
     *    +3 -> _shp_sze: (size)
     *
     *    +4 -> _tln_ptr: for _shape index to _value _translation
     *    +5 -> _tln_sze: (size)
     * */
    public TensorKernel(){
        this.setExplicit(true);
        _pointers = new int[]{};//0, -initialSize, 0, 0, 0, 0
        _shapes = new int[0];
        _translations = new int[0];
        _values = new double[0];
    }
    //-------------------------------------------------------------------
    private int _tsr_count(){
        return _pointers.length/6;
    }
    //-------------------------------------------------------------------
    // Tensors:
    private int _tsr_ptr(int t_id){
        return _pointers[t_id*6+0 ];
    }
    private int _tsr_sze(int t_id){
        return Math.abs(_pointers[t_id*6+1]);
    }
    private int _tsr_end(int t_id){
        return (t_id<0)?0:(_tsr_ptr(t_id)+_tsr_sze(t_id)+ _tsr_grd_sze(t_id));
    }

    private void _set_tsr_ptr(int t_id, int ptr){
        _pointers[t_id*6+0 ] = ptr;
    }
    private void _set_tsr_sze(int t_id, int sze){
        _pointers[t_id*6+1]=sze;
    }

    // Gradients:
    private int _tsr_grd_ptr(int t_id){
        return _tsr_ptr(t_id)+_tsr_sze(t_id);
    }
    private int _tsr_grd_sze(int t_id){return _tsr_sze(t_id)*((_has_gradient(t_id))?1:0);}
    private int _tsr_grd_end(int t_id){return _tsr_grd_ptr(t_id)+_tsr_sze(t_id);}

    // Shapes:
    private int _shp_ptr(int t_id){
        return _pointers[t_id*6+2 ];
    }
    private int _shp_sze(int t_id){
        return _pointers[t_id*6+3];
    }
    private int _shp_end(int t_id){return _shp_ptr(t_id)+ _shp_sze(t_id);}

    private void _set_shp_ptr(int t_id, int ptr){
        _pointers[t_id*6+2 ] = ptr;
    }
    private void _set_shp_sze(int t_id, int sze){
        _pointers[t_id*6+3]=sze;
    }

    // Translations:
    private int _tln_ptr(int t_id){
        return _pointers[t_id*6+4 ];
    }
    private int _tln_sze(int t_id){
        return _pointers[t_id*6+5];
    }
    private int _tln_end(int t_id){return _tln_ptr(t_id)+ _tln_sze(t_id);}

    private void _set_tln_ptr(int t_id, int ptr){
        _pointers[t_id*6+4 ] = ptr;
    }
    private void _set_tln_sze(int t_id, int sze){
        _pointers[t_id*6+5]=sze;}
    //-------------------------------------------------------------------
    private int _free_spc(int t_id){
        return ((t_id+1)*6<_pointers.length)
                ?(_tsr_ptr(t_id+1)-_tsr_end(t_id))//=> getting space between t_id and next element
                :_values.length-_tsr_end(t_id);//=> t_id is last element
    }
    //-------------------------------------------------------------------
    private void _setNull(int t_id){
        _pointers[t_id*6+1]=0;
    }
    private boolean _ptr_is_null(int t_id){
        return (t_id<0)?false:(_pointers[t_id*6+1]==0);
    }
    private boolean _has_gradient(int t_id){
        return (_pointers[t_id*6+1]<0);
    }
    //-------------------------------------------------------------------
    /**
     *    pointer modification (allocation and freeing)
     *    ------------------------------------------------------
     * */
    public int freePtrOf(int t_id, int[][] regis){
        return _mod_ptrs(t_id, true, regis);
    }

    public int allocPtrFor(T tensor, int[][] regis){
        int size = tensor.size();
        int[] shape = tensor.shape();
        int[] translation = tensor.translation();
        __shp = shape;
        __tln = translation;
        this.put(__shp);
        this.put(__tln);
        int biggestChunck = 0;
        for(int t_i = -1; t_i< _tsr_count(); t_i++){
            biggestChunck = (biggestChunck< _free_spc(t_i))? _free_spc(t_i):biggestChunck;
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
        for(int t_i = -1; t_i< _tsr_count(); t_i++){
            if(_free_spc(t_i)>=size){
                int ptr = 0;
                if(_ptr_is_null(t_i)==false){
                    ptr = _mod_ptrs(t_i, false, regis);
                    t_i++;
                }else{
                    regis[0][0] = 0;
                }
                _set_tsr_ptr(t_i, _tsr_end(((t_i>0)?t_i-1:0)));
                _set_tsr_sze(t_i, size*((tensor.rqsGradient())?-1:1));
                _set_shp_ptr(t_i, _alloc_cfg(shape, _shapes, false));
                _set_tln_ptr(t_i, _alloc_cfg(translation, _translations, true));
                _set_shp_sze(t_i, shape.length);
                _set_tln_sze(t_i, translation.length);
                this.put(_pointers);
                return ptr;
            }
        }
        this.put(_pointers);
        return 0;//return pointer f _alloc_val_sizer
    }

    private int _mod_ptrs(int t_id, boolean rmv, int[][] regis){
        int[] mapper = new int[regis[0].length];//_tsr_count()
        int rgr_ptr = 0;
        for(int i=0; i<mapper.length; i++){
            if(regis[0][i]>=0){//=> REGISTER contains t_id's or null pointer (-1)
                mapper[regis[0][i]] = i;//=> mapper points from pointer entries to REGISTER entries
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
                    regis[0][i]=t_id+1;
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
        return rgr_ptr;
    }

    /**
     *    returns pointer to _shapes/_translations elements.
     *    moves _shapes/_translations to device if required data is not present.
     * */
    private int _alloc_cfg(int[] cfg, int[] cfgs, boolean is_tln){
        int zeros_ptr = 0;
        for(int i = 0; i<cfgs.length; i++){
            if(cfgs[i]!=0){
                zeros_ptr = i+1;
            }
            boolean matches = true;
            for(int ii=0; ii<cfg.length; ii++){
                if((i+ii>=cfgs.length)||cfgs[i+ii]!=cfg[ii]&&cfgs[i+ii]!=0){
                    matches=false;
                }
            }
            if(matches){
                for(int ii=0; ii<cfg.length; ii++){
                    cfgs[i+ii]=cfg[ii];
                }
                this.put(cfgs);
                return i;
            }
        }
        int[] new_cfgs
                = new int[
                     cfgs.length +
                     (
                          cfg.length>((int)(cfgs.length* alloc_shp_tln_sizer))
                                  ?cfg.length
                                  :((int)(cfgs.length* alloc_shp_tln_sizer))
                     )
                ];
        for(int i=0; i<zeros_ptr; i++){
            new_cfgs[i] = cfgs[i];
        }
        for(int i=zeros_ptr; i<zeros_ptr+cfg.length; i++){
            new_cfgs[i] = (i<zeros_ptr+cfg.length)?cfg[i-zeros_ptr]:0;
        }
        int ptr = zeros_ptr;
        if(is_tln){
            _translations = new_cfgs;
            this.put(_translations);
        } else {
            _shapes = new_cfgs;
            this.put(_shapes);
        }
        return ptr;
    }

    /**
     *    Pre-Execution functions (mode setter)
     *    return global size for range creation!
     *    ------------------------------------------------------
     *    ======================================================
    * */
    public int executionSizeOf_fetchTsr(int t_id, boolean grd){
        __val = new double[_tsr_sze(t_id)];
        __shp = new int[_shp_sze(t_id)];
        __tln = new int[_tln_sze(t_id)];
        _mde = new int[]{(grd)?-4:-3, t_id};
        this.put(__val).put(__shp).put(__tln).put(_mde);
        int g_sze = _tsr_sze(t_id) + _shp_sze(t_id) + _tln_sze(t_id);
        return g_sze;
    }

    public int executionSizeOf_storeTsr(int t_id, double[] value, boolean grd){
        _mde = new int[]{(grd)?-2:-1, t_id};// 1. define if stored as grd or not; 2. specify tsr id;
        __val = value;
        this.put(_mde).put(__val);
        int g_sze = _tsr_sze(t_id);
        return g_sze;
    }

    /**
     * Mode format:
     * 0: _id
     * 1-(n-1): t_id's
     * n: d (-1 if not derivative)
     * */
    public int executionSizeOf_calc(int[] mode, byte gradPtrMod){// Mode contains _id, drain id and source id's !
        if(_mde ==null||_mde.length<3||_mde.length!=mode.length){
            _mde = mode;
            this._set_grad_ptr(gradPtrMod);
            this.put(_mde);//up
        }
        for(int i = 0; i< _idx.length; i++){
            _idx[i] = 0;
        }
        this.put(_idx);
        return _tsr_sze(mode[1]);
    }

    /**
     * Mode format:
     * 0: _id
     * 1-(n-1): t_id's
     * n: d (-1 if not derivative)
     * */
    public int executionSizeOf_calc(int[] mode, double value, byte gradPtrMod)
    {// Mode contains _id, _value is applied to all!
        int size = executionSizeOf_calc(mode, gradPtrMod);
        __val = new double[]{value};
        this.put(__val);
        return size;
    }

    private void _set_grad_ptr(byte flags){
        if(flags!=0){
            for(int i=0; i<_mde.length-1; i++){
                if((flags & (1<<i))==1){
                    _set_tsr_ptr(_mde[i+1],_tsr_ptr(_mde[i+1]) + _tsr_sze(_mde[i+1]));
                }
            }
            this.put(_pointers);
        }
    }

    public void resetGradPtr(byte flags){
        if(flags!=0){
            for(int i=1; i<_mde.length-1; i++){
                if((flags & (1<<(i-1)))==1){
                    _set_tsr_ptr(_mde[i],_tsr_ptr(_mde[i]) - _tsr_sze(_mde[i]));
                }
            }
            this.put(_pointers);
        }
    }

    /**
     *    KERNEL RUN:
     *    ==========
     * */
    @Override
    public void run() {
        _run(this.getGlobalId());
    }

    public void run(int gid, int[] m){//entry point for cpu! (testing)
        _mde = m;
        _run(gid);
    }

    private void _run(int gid)
    {
        for(int i = 0; i< _idx.length; i++){
            _idx[i] = 0;
        }
        if(_mde[0]==-5){//cleanup //TODO implement!
        }
        if(_mde[0]==-4){//executionSizeOf_fetchTsr grad f tensor
            _run_fetch(gid, true);
        }
        if(_mde[0]==-3){//executionSizeOf_fetchTsr tensor
            _run_fetch(gid, false);
        }
        if(_mde[0]==-2){//executionSizeOf_storeTsr grad f tensor
            _run_store(gid, true);
        }
        if(_mde[0]==-1){//executionSizeOf_storeTsr tensor
            _run_store(gid, false);
        }
        if(_mde[0]==0){//Relu
            _run_relu(gid, _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==1){//Sigmoid
            _run_sig(gid, _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==2){//Tanh
            _run_tnh(gid, _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==3){//Quadratic
            _run_qdr(gid, _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==4){//Ligmoid
            _run_lig(gid, _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==5){//Linear
            _run_lin(gid, _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==6){//Gaussian
            _run_gus(gid, _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==7){//Absolut
            _run_abs(gid, _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==8){//Sinus
            _run_sin(gid, _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==9){//Cosinus
            _run_cos(gid, _mde[1], _mde[2], _mde[3]);
        }
        if(_mde[0]==10){//Sum
            _run_sum(gid, _mde[1], (_mde.length<=3)?-1: _mde[_mde.length-1]);
        }
        if(_mde[0]==11){//Product
            _run_pi(gid, _mde[1], (_mde.length<=3)?-1: _mde[_mde.length-1]);
        }
        if(_mde[0]==12){//  ^
            if(_mde.length>2) {
                _run_pow(gid, _mde[1], _mde[2], _mde[3]);
            } else {
                _run_broadcast_pow(gid, _mde[1], __val[0]);
            }
        }
        if(_mde[0]==13){//  /
            if(_mde.length>2){
                _run_div(gid, _mde[1], _mde[2], _mde[3]);
            } else {
                _run_broadcast_div(gid, _mde[1], __val[0]);
            }
        }
        if(_mde[0]==14){//  *
            if(_mde.length>2) {
                _run_mul(gid, _mde[1], _mde[2], _mde[3]);
            }else{
                _run_broadcast_mul(gid, _mde[1], __val[0]);
            }
        }
        if(_mde[0]==15){//  %
            if(_mde.length>2) {
                _run_mod(gid, _mde[1], _mde[2], _mde[3]);
            } else{
                _run_broadcast_mod(gid, _mde[1], __val[0]);
            }
        }
        if(_mde[0]==16){//  -
            if(_mde.length>2) {
                _run_sub(gid, _mde[1], _mde[2], _mde[3]);
            }else{
                _run_broadcast_sub(gid, _mde[1], __val[0]);
            }
        }
        if(_mde[0]==17){//  +
            if(_mde.length>2) {
                _run_add(gid, _mde[1], _mde[2], _mde[3]);
            } else {
                _run_broadcast_add(gid, _mde[1], __val[0]);
            }
        }
        if(_mde[0]==18){// x
            if(_mde[_mde.length-1]<0){
                _run_conv(gid, _mde[1], _mde[2], _mde[3]);
            } else {
                _run_conv_inv(gid, _mde[1], _mde[2], _mde[3], (_mde[_mde.length-1]==0));
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
    private void _run_cleanup(int gid){
        //TODO: implement
        //TODO write test cases!
    }

    private void _run_fetch(int gid, boolean grd){
        if(gid<_tsr_sze(_mde[1])){
            __val[gid] = _values[((!grd)?_tsr_ptr(_mde[1]):_tsr_grd_ptr(_mde[1]))+gid];
        }else{
            if(gid<(_tsr_sze(_mde[1]) + _shp_sze(_mde[1]))){
                gid -= _tsr_sze(_mde[1]);
                __shp[gid] = _shapes[_shp_ptr(_mde[1])+gid];
            }else{
                if(gid<(_tsr_sze(_mde[1]) + _shp_sze(_mde[1]) + _tln_sze(_mde[1]))){
                    gid -= (_tsr_sze(_mde[1]) + _shp_sze(_mde[1]));
                    __tln[gid] = _translations[_tln_ptr(_mde[1])+gid];
                }
            }
        }
    }

    private void _run_store(int gid, boolean grd){
        if(gid<_tsr_sze(_mde[1])){
            _values[((!grd)?_tsr_ptr(_mde[1]):_tsr_grd_ptr(_mde[1]))+gid]=__val[gid];
        }else{
            if(gid<(_tsr_sze(_mde[1])+ _shp_sze(_mde[1]))){
                gid -= _tsr_sze(_mde[1]);
                _shapes[_shp_ptr(_mde[1])+gid]=__shp[gid];
            }else{
                if(gid<(_tsr_sze(_mde[1])+ _shp_sze(_mde[1])+ _tln_sze(_mde[1]))){
                    gid -= (_tsr_sze(_mde[1])+ _shp_sze(_mde[1]));
                    _translations[_tln_ptr(_mde[1])+gid]=__tln[gid];
                }
            }
        }
    }

    private void _run_relu(int gid, int drn_id, int src_id, int d){
        if (d<0) {
            if (_values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)] >= 0) {
                _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                        (_values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)]);
            } else {
                _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                        (_values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)]) * 0.01;
            }
        } else {
            if (_values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)] >= 0) {
                _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = 0.01;
            } else {
                _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = 0.01;
            }
        }
    }

    private void _run_sig(int gid, int drn_id, int src_id, int d){
        if(d<0){
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    1 / (1 + Math.pow(Math.E, (-_values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)])));

        }else{
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                (
                    Math.pow(
                        Math.E,
                        -_values[_tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)]
                    )
                ) / (Math.pow(
                        (1 + Math.pow(
                            Math.E,
                            -_values[_tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)]
                        )
                    ), 2)
                        +
                        2 * Math.pow(Math.E, -_values[_tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)])
                );
        }
    }

    private void _run_tnh(int gid, int drn_id, int src_id, int d){
        if(d<0){
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    _values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                            / Math.pow(
                                    (1 + Math.pow(
                                            _values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                                            , 2)
                                    ), 0.5);

        }else{
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                (1 - Math.pow(
                    (_values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                        /
                        Math.pow(
                            (1 + Math.pow(
                                _values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                                , 2)
                            ), 0.5
                        )
                    ), 2)
                );

        }
    }
    private void _run_qdr(int gid, int drn_id, int src_id, int d){
        if(d<0){
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.pow(_values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)],2);
        }else{
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    _values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)]*2;
        }
    }
    private void _run_lig(int gid, int drn_id, int src_id, int d){
        if(d<0){
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = (
                Math.log(1+Math.pow(Math.E, _values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)]))
            );
        }else{
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                1 /
                    (1 + Math.pow(
                        Math.E,
                        _values[_tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)]
                    )
                );
        }
    }

    private void _run_lin(int gid, int drn_id, int src_id, int d){
        if(d<0){
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    _values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)];
        }else{
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    _values[_tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)];
        }
    }

    private void _run_gus(int gid, int drn_id, int src_id, int d){
        if(d<0){
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    Math.pow(Math.E, -Math.pow(_values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)], 2));
        }else{
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    -2 * (_values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)])
                            * Math.pow(Math.E, -Math.pow(_values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)], 2));
        }
    }

    private void _run_abs(int gid, int drn_id, int src_id, int d){
        _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.abs(_values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)]);
    }
    private void _run_sin(int gid, int drn_id, int src_id, int d){
        _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.sin(_values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)]);
    }
    private void _run_cos(int gid, int drn_id, int src_id, int d){
        _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.cos(_values[_tsr_ptr(src_id)+__i_of(gid, src_id, 1)]);
    }

    private void _run_sum(int gid, int drn_id, int d){
        if(d<0){
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = 0;
            for(int i = 2; i<(_mde.length-1); i++){
                _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] +=
                        _values[_tsr_ptr(_mde[i])+__i_of(gid, _mde[i], 1)];
            }
        }else{
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    _values[_tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)];
        }
    }

    private void _run_pi(int gid, int drn_id, int d){
        if(d<0){
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = 0;
            for(int i = 2; i<(_mde.length-1); i++){
                _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] *=
                        _values[_tsr_ptr(_mde[i])+__i_of(gid, _mde[i], 1)];
            }
        }else{
            //TODO: implement ...............
            _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    _values[_tsr_ptr(_mde[2+d])+__i_of(gid, _mde[2+d], 1)];//........
        }
    }

    private void _run_pow(int gid, int drn_id, int src1_id, int src2_id){
        _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.pow(
                        _values[_tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)],
                        _values[_tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)]
                        );
    }
    private void _run_broadcast_pow(int gid, int drn_id, double value){
        _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.pow(_values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)], value);
    }
    private void _run_div(int gid, int drn_id, int src1_id, int src2_id){
        _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                _values[_tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)]
                        /
                _values[_tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)];
    }
    private void _run_broadcast_div(int gid, int drn_id, double value){
        _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)]/value;
    }
    private void _run_mul(int gid, int drn_id, int src1_id, int src2_id){
        _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                _values[_tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)]
                        *
                _values[_tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)];
    }
    private void _run_broadcast_mul(int gid, int drn_id, double value){
        _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)]*value;
    }
    private void _run_mod(int gid, int drn_id, int src1_id, int src2_id){
        _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                ((int)_values[_tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)])
                        %
                ((int)_values[_tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)]);
    }
    private void _run_broadcast_mod(int gid, int drn_id, double value){
        _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                (int)(_values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)])%(int)value;
    }
    private void _run_sub(int gid, int drn_id, int src1_id, int src2_id){
        _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                _values[_tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)]
                        -
                _values[_tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)];
    }
    private void _run_broadcast_sub(int gid, int drn_id, double value){
        _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)]
                = _values[_tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)]-value;
    }
    private void _run_add(int gid, int drn_id, int src1_id, int src2_id){
        int i1 = _tsr_ptr(drn_id)+__i_of(gid, drn_id, 0);
        int i2 = _tsr_ptr(src1_id)+__i_of(gid, src1_id, 1);
        int i3 = _tsr_ptr(src2_id)+__i_of(gid, src2_id, 2);
        _values[i1] = _values[i2] + _values[i3];
    }
    private void _run_broadcast_add(int gid, int drn_id, double value){
        int i1 = _tsr_ptr(drn_id)+__i_of(gid, drn_id, 0);
        _values[i1] = _values[i1]+value;
    }
    //==================================================================================================================
    private void _run_conv_inv(int gid, int drn_id, int src1_id, int src2_id, boolean first){
        drn_id = drn_id^src2_id; src2_id = drn_id^src2_id; drn_id = drn_id^src2_id;
        if(first){
            src1_id = src1_id^drn_id; drn_id = src1_id^drn_id; src1_id = src1_id^drn_id;
        }
        // SETUP:
        int p_data_src1 = _tsr_ptr(src1_id);
        int p_data_src2 = _tsr_ptr(src2_id);
        int p_data_drn = _tsr_ptr(drn_id);

        int p_shp_src1 = _shp_ptr(src1_id);
        int p_shp_src2 = _shp_ptr(src2_id);
        int p_shp_drn  = _shp_ptr(drn_id);

        int p_tln_src1 = _tln_ptr(src1_id);
        int p_tln_src2 = _tln_ptr(src2_id);
        int p_tln_drn  = _tln_ptr(drn_id);

        int rank = _shp_sze(drn_id);
        int p_idx_src1 = 0*rank;
        int p_idx_src2 = 1*rank;
        int p_idx_drn  = 2*rank;

        int src1End = p_shp_src1 + rank;
        int src2End = p_shp_src2 + rank;

        //increment on drain:
        for(int i=0; i<gid; i++){
            __increment_idx(p_shp_drn, p_idx_drn, rank);
        }
        //increment src accordingly:
        int ri = 0;
        while (ri < rank) {
            if (_idx[(p_idx_src2+ri)] == _shapes[(p_shp_src2+ri)]) {
                _idx[(p_idx_src1 + ri)] = _idx[(p_idx_drn + ri)];
                _idx[(p_idx_src2 + ri)] = 0;
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
                    int i1 = __i_of_idx_on_tln(p_tln_src1, p_idx_src1, rank);
                    int i2 = __i_of_idx_on_tln(p_tln_src2, p_idx_src2, rank);
                    value += _values[(p_data_src1 + i1)] * _values[(p_data_src2 + i2)];
                }
                incrementing = true;
                ri=0;
            } else {//incrementing:
                if (_idx[(p_idx_src2+ri)] < _shapes[(p_shp_src2+ri)]) {
                    _idx[(p_idx_src2+ri)]++;
                    if (_idx[(p_idx_src2+ri)] == _shapes[(p_shp_src2+ri)]) {
                        if (((p_shp_src2+ri) == (src2End - 1))) {
                            running = false;
                        }
                        _idx[(p_idx_src1+ri)] = _idx[(p_idx_drn+ri)];
                        _idx[(p_idx_src2+ri)] = 0;
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
        }
        //set _value in drn:
        int di = __i_of_idx_on_tln(p_tln_drn, p_idx_drn, rank);
        _values[(p_data_drn + di)] = value;
    }//=================================================================================================================

    private void _run_conv(int gid, int drn_id, int src1_id, int src2_id){
        // SETUP:
        int p_data_src1 = _tsr_ptr(src1_id);
        int p_data_src2 = _tsr_ptr(src2_id);
        int p_data_drn = _tsr_ptr(drn_id);

        int p_shp_src1 = _shp_ptr(src1_id);
        int p_shp_src2 = _shp_ptr(src2_id);
        int p_shp_drn  = _shp_ptr(drn_id);

        int p_tln_src1 = _tln_ptr(src1_id);
        int p_tln_src2 = _tln_ptr(src2_id);
        int p_tln_drn  = _tln_ptr(drn_id);

        int rank = _shp_sze(drn_id);
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
                int i1 = __i_of_idx_on_tln(p_tln_src1, p_idx_src1, rank);
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
        //set _value in drn:
        int di = __i_of_idx_on_tln(p_tln_drn, p_idx_drn, rank);
        _values[(p_data_drn + di)] = value;
    }

    //Helper methods for tsr conv:
    private int __increment_At(int ri, int idx_ptr, int shp_ptr) {
        if (_idx[idx_ptr+ri] < (_shapes[shp_ptr+ri])) {
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
        while (ri >= 0 && ri < rank) {
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
        int p_shp  = _shp_ptr(t_id);
        int p_tln  = _tln_ptr(t_id);
        int rank   = _shp_sze(t_id);
        int p_idx  = rank * num;
        for(int i=0; i<gid; i++){
            __increment_idx(p_shp, p_idx, rank);
        }
        return __i_of_idx_on_tln(p_tln, p_idx, rank);
    }

}
