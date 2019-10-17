package neureka.core.device.aparapi;

import com.aparapi.Kernel;
import neureka.core.Tsr;

public abstract class AbstractKernel extends Kernel {

    protected int _biggest_free = 0;
    protected float _free_ratio = -1;
    protected static float _unalloc_limit = 1; //  unalloc/alloc should mot be greater than 1
    protected static float _alloc_val_sizer = 1.5f;
    protected static float alloc_shp_tln_sizer = 1.2f;

    //private void _calc_biggest_free(){
    //    int biggest = 0;
    //    for(int t_i = -1; t_i< _tsr_count(); t_i++){
    //        biggest = (biggest <_free_spc(t_i)) ? _free_spc(t_i):biggest;
    //    }
    //    _biggest_free = biggest;
    //}
    //private void _calc_free_ratio(){
    //    float alloc = 0;
    //    float free = this._free_spc(-1);
    //    for(int i = 0; i< _tsr_count(); i++){
    //        alloc += this._tsr_grd_end(i)-this._tsr_ptr(i);
    //        free += this._free_spc(i);
    //    }
    //    this._free_ratio =  (free/alloc);
    //}

    //------------------------------------------

    protected abstract void _put_new_val(double[] newVal);
    protected abstract void _put_new_val(int newValSize);

    protected int[] __shp;
    protected int[] __tln;
    //-----------------------------
    public @PrivateMemorySpace(8*2) int[] _idx = new int[8*2];
    //-----------------------------
    public int[] _mde = {0};

    /**    TENSORS:   * */
    /* public float/double[] _values; *///<= must be implemented by child classes
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
    //-------------------------------------------------------------------
    protected int _tsr_count(){
        return _pointers.length/6;
    }
    //-------------------------------------------------------------------
    // Tensors:
    protected int _tsr_ptr(int t_id){
        return _pointers[t_id*6+0 ];
    }
    protected int _tsr_sze(int t_id){
        return Math.abs(_pointers[t_id*6+1]);
    }
    protected int _tsr_end(int t_id){
        return (t_id<0)?0:(_tsr_ptr(t_id)+_tsr_sze(t_id)+ _tsr_grd_sze(t_id));
    }

    protected void _set_tsr_ptr(int t_id, int ptr){
        _pointers[t_id*6+0 ] = ptr;
    }
    protected void _set_tsr_sze(int t_id, int sze){
        _pointers[t_id*6+1]=sze;
    }

    // Gradients:
    protected int _tsr_grd_ptr(int t_id){
        return _tsr_ptr(t_id)+_tsr_sze(t_id);
    }
    protected int _tsr_grd_sze(int t_id){return _tsr_sze(t_id)*((_has_gradient(t_id))?1:0);}
    protected int _tsr_grd_end(int t_id){return _tsr_grd_ptr(t_id)+_tsr_sze(t_id);}

    // Shapes:
    protected int _shp_ptr(int t_id){
        return _pointers[t_id*6+2 ];
    }
    protected int _shp_sze(int t_id){
        return _pointers[t_id*6+3];
    }
    protected int _shp_end(int t_id){return _shp_ptr(t_id)+ _shp_sze(t_id);}

    protected void _set_shp_ptr(int t_id, int ptr){
        _pointers[t_id*6+2 ] = ptr;
    }
    protected void _set_shp_sze(int t_id, int sze){
        _pointers[t_id*6+3]=sze;
    }

    // Translations:
    protected int _tln_ptr(int t_id){
        return _pointers[t_id*6+4 ];
    }
    protected int _tln_sze(int t_id){
        return _pointers[t_id*6+5];
    }
    protected int _tln_end(int t_id){return _tln_ptr(t_id)+ _tln_sze(t_id);}

    protected void _set_tln_ptr(int t_id, int ptr){
        _pointers[t_id*6+4 ] = ptr;
    }
    protected void _set_tln_sze(int t_id, int sze){
        _pointers[t_id*6+5]=sze;}
    //-------------------------------------------------------------------
    protected int _free_spc(int t_id){
        return ((t_id+1)*6<_pointers.length)
                ?(_tsr_ptr(t_id+1)-_tsr_end(t_id))//=> getting space between t_id and next element
                :_value_length()-_tsr_end(t_id);//=> t_id is last element
    }
    //-------------------------------------------------------------------
    protected void _setNull(int t_id){
        _pointers[t_id*6+1]=0;
    }
    protected boolean _ptr_is_null(int t_id){
        return (t_id<0)?false:(_pointers[t_id*6+1]==0);
    }
    protected boolean _has_gradient(int t_id){
        return (_pointers[t_id*6+1]<0);
    }
    //-------------------------------------------------------------------

    public abstract double[] values();

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
    public abstract double[] value();
    protected abstract int _value_length();

    protected abstract void _resize_values(int newSize);

    public int[] shape(){this.get(__shp); return __shp;}
    public int[] translation(){this.get(__tln); return __tln;}

    /**
     *    pointer modification (allocation and freeing)
     *    ------------------------------------------------------
     * */
    public int freePtrOf(int t_id, int[][] regis){
        return _mod_ptrs(t_id, true, regis);
    }

    public int allocPtrFor(Tsr tensor, int[][] regis){
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
            int newSpace = (int)(_value_length()* _alloc_val_sizer);
            newSpace = (newSpace>size)?newSpace:size;
            _resize_values(_value_length()+newSpace);
            if(_pointers.length==0){
                _pointers = new int[]{0, 0, 0, 0, 0, 0};
            }
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

    protected int _mod_ptrs(int t_id, boolean rmv, int[][] regis){
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
    protected int _alloc_cfg(int[] cfg, int[] cfgs, boolean is_tln){
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

    protected void _set_grad_ptr(byte flags){
        if(flags!=0){
            for(int i=0; i<_mde.length-1; i++){
                if((flags & (1<<i))==1){
                    _set_tsr_ptr(_mde[i+1],_tsr_ptr(_mde[i+1]) + _tsr_sze(_mde[i+1]));
                }
            }
            this.put(_pointers);
        }
    }

    /**
     *    Pre-Execution functions (mode setter)
     *    return global size for range creation!
     *    ------------------------------------------------------
     *    ======================================================
     * */
    public int executionSizeOf_fetchTsr(int t_id, boolean grd){
        //__val = new double[_tsr_sze(t_id)];
        _put_new_val(_tsr_sze(t_id));
        __shp = new int[_shp_sze(t_id)];
        __tln = new int[_tln_sze(t_id)];
        _mde = new int[]{(grd)?-4:-3, t_id};
        this.put(__shp).put(__tln).put(_mde);//.put(__val);
        int g_sze = _tsr_sze(t_id) + _shp_sze(t_id) + _tln_sze(t_id);
        return g_sze;
    }

    public int executionSizeOf_storeTsr(int t_id, double[] value, boolean grd){
        _mde = new int[]{(grd)?-2:-1, t_id};// 1. define if stored as grd or not; 2. specify tsr id;
        _put_new_val(value);
        //__val = value;
        this.put(_mde);//.put(__val);
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
        //__val = new double[]{value};
        //this.put(__val);
        _put_new_val(new double[]{value});
        return size;
    }

    public void closeExecution(byte gradPtrMod){
        if(gradPtrMod !=0){
            for(int i=1; i<_mde.length-1; i++){
                if((gradPtrMod & (1<<(i-1)))==1){
                    _set_tsr_ptr(_mde[i],_tsr_ptr(_mde[i]) - _tsr_sze(_mde[i]));
                }
            }
            this.put(_pointers);
        }
    }

    public void run(int gid, int[] m){//entry point for cpu! (testing)
        _mde = m;
        _run(gid);
    }


    @Override
    public abstract void run();


    protected void _run(int gid)
    {
        for(int i = 0; i< _idx.length; i++){
            _idx[i] = 0;
        }
        if(_mde[0]==-5){//cleanup //TODO implement!
        }
        if(_mde[0]==-4){//executionSizeOf_fetchTsr grad of tensor
            _run_fetch(gid, true);
        }
        if(_mde[0]==-3){//executionSizeOf_fetchTsr tensor
            _run_fetch(gid, false);
        }
        if(_mde[0]==-2){//executionSizeOf_storeTsr grad of tensor
            _run_store(gid, true);
        }
        if(_mde[0]==-1){//executionSizeOf_storeTsr tensor
            _run_store(gid, false);
        }
        if(_mde[0]==0){//Relu
            _run_relu(gid);
        }
        if(_mde[0]==1){//Sigmoid
            _run_sig(gid);
        }
        if(_mde[0]==2){//Tanh
            _run_tnh(gid);
        }
        if(_mde[0]==3){//Quadratic
            _run_qdr(gid);
        }
        if(_mde[0]==4){//Ligmoid
            _run_lig(gid);
        }
        if(_mde[0]==5){//Linear
            _run_lin(gid);
        }
        if(_mde[0]==6){//Gaussian
            _run_gus(gid);
        }
        if(_mde[0]==7){//Absolut
            _run_abs(gid);
        }
        if(_mde[0]==8){//Sinus
            _run_sin(gid);
        }
        if(_mde[0]==9){//Cosinus
            _run_cos(gid);
        }
        if(_mde[0]==10){//Sum
            _run_sum(gid);
        }
        if(_mde[0]==11){//Product
            _run_pi(gid);
        }
        if(_mde[0]==12){//  ^
            if(_mde.length>2) {
                _run_pow(gid);
            } else {
                _run_broadcast_pow(gid);
            }
        }
        if(_mde[0]==13){//  /
            if(_mde.length>2){
                _run_div(gid);
            } else {
                _run_broadcast_div(gid);
            }
        }
        if(_mde[0]==14){//  *
            if(_mde.length>2) {
                _run_mul(gid);
            }else{
                _run_broadcast_mul(gid);
            }
        }
        if(_mde[0]==15){//  %
            if(_mde.length>2) {
                _run_mod(gid);
            } else{
                _run_broadcast_mod(gid);
            }
        }
        if(_mde[0]==16){//  -
            if(_mde.length>2) {
                _run_sub(gid);
            }else{
                _run_broadcast_sub(gid);
            }
        }
        if(_mde[0]==17){//  +
            if(_mde.length>2) {
                _run_add(gid);
            } else {
                _run_broadcast_add(gid);
            }
        }
        if(_mde[0]==18){// x
            _run_conv(gid);
        }

    }

    protected abstract void _run_fetch(int gid, boolean grd);

    protected abstract void _run_store(int gid, boolean grd);

    protected abstract void _run_relu(int gid);

    protected abstract void _run_sig(int gid);

    protected abstract void _run_tnh(int gid);

    protected abstract void _run_qdr(int gid);

    protected abstract void _run_lig(int gid);

    protected abstract void _run_lin(int gid);

    protected abstract void _run_gus(int gid);

    protected abstract void _run_abs(int gid);

    protected abstract void _run_sin(int gid);

    protected abstract void _run_cos(int gid);

    protected abstract void _run_sum(int gid);

    protected abstract void _run_pi(int gid);

    protected abstract void _run_pow(int gid);

    protected abstract void _run_broadcast_pow(int gid);

    protected abstract void _run_div(int gid);

    protected abstract void _run_broadcast_div(int gid);

    protected abstract void _run_mul(int gid);

    protected abstract void _run_broadcast_mul(int gid);

    protected abstract void _run_mod(int gid);

    protected abstract void _run_broadcast_mod(int gid);

    protected abstract void _run_sub(int gid);

    protected abstract void _run_broadcast_sub(int gid);

    protected abstract void _run_add(int gid);

    protected abstract void _run_broadcast_add(int gid);

    //==================================================================================================================
    protected abstract void _run_conv(int gid);
    
    //=================================================================================================================
    //Helper methods for execution:
    //----------------------------
    protected int __i(int gid, int m){
        return _tsr_ptr(_mde[m])+ __i_of_idx_on_shp(gid, _mde[m], m-1);
    }

    protected int __d(){
        return _mde[_mde.length-1];
    }

    protected int __n(){
        return _mde.length-2;
    }

    protected void __gid_to_idx(int gid, int idx_ptr, int tln_ptr, int rank){
        int ri = rank;
        do {
            ri--;
            _idx[idx_ptr+ri] = gid/_translations[tln_ptr+ri];
            gid = gid % _translations[tln_ptr+ri];
        } while (ri > 0);
    }

    protected int __increment_At(int ri, int idx_ptr, int shp_ptr) {
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
    protected void __increment_idx(int shp_ptr, int idx_ptr, int rank) {
        int ri = 0;
        while (ri >= 0 && ri < rank) {
            ri = __increment_At(ri, idx_ptr, shp_ptr);
        }
    }

    protected int __i_of_idx_on_tln(int p_tln, int idx_ptr, int rank) {
        int i = 0;
        for (int ii = 0; ii < rank; ii++) {
            i += _translations[p_tln+ii] * _idx[idx_ptr+ii];
        }
        return i;
    }

    protected int __i_of_idx_on_shp(int gid, int t_id, int num){
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
