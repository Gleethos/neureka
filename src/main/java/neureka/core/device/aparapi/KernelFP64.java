package neureka.core.device.aparapi;

public class KernelFP64 extends AbstractKernel
{
    /**   +============================+
     *    || DEFINING FP - TYPE (64): ||
     *    +============================+
     * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

    public KernelFP64(){
        this.setExplicit(true);
        _pointers = new int[]{};//0, -initialSize, 0, 0, 0, 0
        _shapes = new int[0];
        _translations = new int[0];
        _values = new double[0];
    }

    /**
     *    SCALAR/IO-VALUE STORAGE:
     *    =======================
     * */
    protected double[] __val;

    @Override
    public double[] value(){this.get(__val); return __val; }

    @Override
    protected void _put_new_val(double[] newVal){
        __val = newVal;
        this.put(__val);
    }

    @Override
    protected void _put_new_val(int newValSize){
        __val = new double[newValSize];
        this.put(__val);
    }

    //---------------------------------------------
    /**
     *    TENSOR VALUES:
     *    =============
     * */
    public double[] _values;

    //---------------------------------------------
    @Override
    public double[] values(){
        this.get(_values);
        return _values;
    }

    @Override
    protected  int _value_length(){
        return _values.length;
    }

    @Override
    protected void _resize_values(int newSize){
        this.get(_values);//TODO: Make a flag so that this is avoided!
        double[] newValues = new double[newSize];
        for(int i = 0; i< _value_length(); i++){
            newValues[i] = _values[i];
        }
        _values = newValues;
        this.put(_values);
    }


    //------------------------------------------

    /**
     *    KERNEL RUN:
     *    ==========
     * */
    @Override
    public void run() {
        _run(this.getGlobalId());
    }

    /*************************************************************************************************************/

    @Override
    protected void _run_fetch(int gid, boolean grd){
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

    @Override
    protected void _run_store(int gid, boolean grd){
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

    @Override
    protected void _run_relu(int gid){
        if (__d()<0) {
            if (_values[__i(gid, 2)] >= 0) {
                _values[__i(gid, 1)] = (_values[__i(gid, 2)]);
            } else {
                _values[__i(gid, 1)] = (_values[__i(gid, 2)]) * 0.01;
            }
        } else {
            if (_values[__i(gid, 2)] >= 0) {
                _values[__i(gid, 1)] = 0.01;
            } else {
                _values[__i(gid, 1)] = 0.01;
            }
        }
    }

    @Override
    protected void _run_sig(int gid){
        if(__d()<0){
            _values[__i(gid, 1)] =
                    1 / (1 + Math.pow(Math.E, (-_values[__i(gid, 2)])));

        }else{
            _values[__i(gid, 1)] =
                (
                    Math.pow(
                        Math.E,
                        -_values[__i(gid, 2)]
                    )
                ) / (Math.pow(
                        (1 + Math.pow(
                            Math.E,
                            -_values[__i(gid, 2)]
                        )
                    ), 2)
                        +
                        2 * Math.pow(Math.E, -_values[__i(gid, 2)])
                );
        }
    }

    @Override
    protected void _run_tnh(int gid){
        if(__d()<0){
            _values[__i(gid, 1)] =
                    _values[__i(gid, 2)]
                            / Math.pow(
                                    (1 + Math.pow(
                                            _values[__i(gid, 2)]
                                            , 2)
                                    ), 0.5);

        }else{
            _values[__i(gid, 1)] =
                (1 - Math.pow(
                    (_values[__i(gid, 2)]
                        /
                        Math.pow(
                            (1 + Math.pow(
                                _values[__i(gid, 2)]
                                , 2)
                            ), 0.5
                        )
                    ), 2)
                );

        }
    }

    @Override
    protected void _run_qdr(int gid){
        if(__d()<0){
            _values[__i(gid, 1)] =
                Math.pow(_values[__i(gid, 2)],2);
        }else{
            _values[__i(gid, 1)] =
                    _values[__i(gid, 2)]*2;
        }
    }

    @Override
    protected void _run_lig(int gid){
        if(__d()<0){
            _values[__i(gid, 1)] = (
                Math.log(1+Math.pow(Math.E, _values[__i(gid, 2)]))
            );
        }else{
            _values[__i(gid, 1)] =
                1 /
                    (1 + Math.pow(
                        Math.E,
                        _values[__i(gid, 2)]
                    )
                );
        }
    }

    @Override
    protected void _run_lin(int gid){
        if(__d()<0){
            _values[__i(gid, 1)] =
                    _values[__i(gid, 2)];
        }else{
            _values[__i(gid, 1)] = 1;
        }
    }

    @Override
    protected void _run_gus(int gid){
        if(__d()<0){
            _values[__i(gid, 1)] =
                    Math.pow(Math.E, -Math.pow(_values[__i(gid, 2)], 2));
        }else{
            _values[__i(gid, 1)] =
                    -2 * (_values[__i(gid, 2)])
                            * Math.pow(Math.E, -Math.pow(_values[__i(gid, 2)], 2));
        }
    }

    @Override
    protected void _run_abs(int gid){
        _values[__i(gid, 1)] =
                Math.abs(_values[__i(gid, 2)]);
    }

    @Override
    protected void _run_sin(int gid){
        _values[__i(gid, 1)] =
                Math.sin(_values[__i(gid, 2)]);
    }

    @Override
    protected void _run_cos(int gid){
        _values[__i(gid, 1)] =
                Math.cos(_values[__i(gid, 2)]);
    }

    @Override
    protected void _run_sum(int gid){
        if(__d()<0){
            _values[__i(gid, 1)] = 0;
            for(int i = 2; i<(_mde.length-1); i++){
                _values[__i(gid, 1)] +=
                        _values[_tsr_ptr(_mde[i])+ __i_of_idx_on_shp(gid, _mde[i], 1)];
            }
        }else{
            _values[__i(gid, 1)] = 1;//_values[_tsr_ptr(_mde[2+d])+__i_of_idx_on_shp(gid, _mde[2+d], 1)];
        }
    }

    @Override
    protected void _run_pi(int gid){
        if(__d()<0){
            _values[__i(gid, 1)] = 1;
            for(int i = 2; i<(_mde.length-1); i++){
                _values[__i(gid, 1)] *=
                        _values[_tsr_ptr(_mde[i])+ __i_of_idx_on_shp(gid, _mde[i], 1)];
            }
        }else{
            //TODO: implement ...............
            _values[__i(gid, 1)] = 666;//_values[_tsr_ptr(_mde[2+d])+__i_of_idx_on_shp(gid, _mde[2+d], 1)];//........
        }
    }

    @Override
    protected void _run_pow(int gid){
        if (__d() < 0) {
            _values[__i(gid, 1)] = _values[__i(gid, 2)];
            for (int i = 2; i < __n(); i++) {
                _values[__i(gid, 1)] = Math.pow(_values[__i(gid, 1)], _values[__i(gid, 1+i)]);
            }
            /**Note:
             * the right side (of x) can be simplified by multiplying!
             * The formular always looks like this:  a^(x*b)
             * **/
        } else {
                double out = 0;
                double b = 1;
                for (int i = 2; i < __n(); i++) {
                    b *= (((i-1)==__d())?1:_values[__i(gid, i+1)]);
                }
                double a = _values[__i(gid, 2)];
                for(int si=0; si<__n(); si++){
                    if(si==0){
                        out += (__d()==0)?a*b*Math.pow(a, b-1):0;
                    } else {
                        out += (a>=0 && __d()==si)?_values[__i(gid, 2+si)]*b*Math.log(a):0;
                    }
                }
                _values[__i(gid, 1)] = out;
        }
    }

    @Override
    protected void _run_broadcast_pow(int gid){
        _values[__i(gid, 1)] =
                Math.pow(_values[__i(gid, 1)], __val[0]);
    }

    @Override
    protected void _run_div(int gid){
        double result = 1;
        if(__d()!=0){
            result = _values[__i(gid, 2)];
        }
        if (__d() < 0) {
            for (int i = 2; i < __n(); i++) {
                result = result / _values[__i(gid, 1+i)];
            }
            _values[__i(gid, 1)] = result;
        } else {
            double temp = 0;
            for (int i = 2; i < __n(); i++) {
                if(__d() > i){
                    result = result / _values[__i(gid, 1+i)];
                } else if (__d() > i){
                    result = result / _values[__i(gid, 1+i)];
                } else {
                    temp = _values[__i(gid, 1+i)];//result;
                    //result = 1;//TODO: Swapping?
                }
            }
            _values[__i(gid, 1)] = result * -Math.pow(temp, -2);
        }


    }

    @Override
    protected void _run_broadcast_div(int gid){
        _values[__i(gid, 1)] = _values[__i(gid, 1)]/__val[0];
    }

    @Override
    protected void _run_mul(int gid){
        double result = 1;
        if(__d()!=0){
            result = _values[__i(gid, 2)];
        }
        if (__d() < 0) {
            for (int i = 2; i < __n(); i++) {
                result = result * _values[__i(gid, 1+i)];
            }
        } else {
            for (int i = 2; i < __n(); i++) {
                double current = 1;
                if(__d() != i){
                      current = _values[__i(gid, 1+i)];
                }
                result = result * current;
            }
        }
        _values[__i(gid, 1)] = result;
    }

    @Override
    protected void _run_broadcast_mul(int gid){
        _values[__i(gid, 1)] = _values[__i(gid, 1)]*__val[0];
    }

    @Override
    protected void _run_mod(int gid){
        _values[__i(gid, 1)] = ((int)_values[__i(gid, 2)]) % ((int)_values[__i(gid, 3)]);
    }

    @Override
    protected void _run_broadcast_mod(int gid){
        _values[__i(gid, 1)] = (int)(_values[__i(gid, 1)])%(int)__val[0];
    }

    @Override
    protected void _run_sub(int gid){
        int i1 = __i(gid, 1);
        if(__d()<0){
            int i2 = __i(gid, 2);
            int i3 = __i(gid, 3);
            _values[i1] = _values[i2] - _values[i3];
        } else {
            _values[i1] = 1;
        }
    }

    @Override
    protected void _run_broadcast_sub(int gid){
        _values[__i(gid, 1)]
                = _values[__i(gid, 1)]-__val[0];
    }

    @Override
    protected void _run_add(int gid){
        if(__d()<0){
            _values[__i(gid, 1)] = _values[__i(gid, 2)] + _values[__i(gid, 3)];
        } else {
            _values[__i(gid, 1)] = 1;
        }
    }

    @Override
    protected void _run_broadcast_add(int gid){
        int i1 = __i(gid, 1);
        _values[i1] = _values[i1] + __val[0];
    }

    @Override
    protected void _run_conv(int gid){///Lets get going!!
        int drn_id = _mde[1];
        int src1_id = _mde[2];
        int src2_id = _mde[3];
        if(__d()>=0) {
            drn_id = drn_id ^ src2_id;
            src2_id = drn_id ^ src2_id;
            drn_id = drn_id ^ src2_id;
            if (__d()==0) {
                src1_id = src1_id ^ drn_id;
                drn_id = src1_id ^ drn_id;
                src1_id = src1_id ^ drn_id;
            }
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
        //__gid_to_idx(gid, p_idx_drn, p_tln_drn, rank);

        //increment src accordingly:
        int ri = 0;
        if(__d() >= 0){
            while (ri < rank) {
                if (_idx[(p_idx_src2+ri)] == _shapes[(p_shp_src2+ri)]) {
                    _idx[(p_idx_src1 + ri)] = _idx[(p_idx_drn + ri)];
                    _idx[(p_idx_src2 + ri)] = 0;
                } else {
                    if (_shapes[(p_shp_drn+ri)] > _shapes[(p_shp_src1+ri)]) {
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
        } else {// conv
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

    }

}
