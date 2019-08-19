package neureka.core.device;
import com.aparapi.Kernel;
import neureka.core.T;
import neureka.core.function.TFunction;

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
    private static float alloc_val_sizer = 1.5f;
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
        this.get(this.values);
        return this.values;
    }
    public int[] shapes(){
        this.get(this.shapes);//probably obsolete
        return this.shapes;
    }
    public int[] translations(){
        this.get(this.translations);//probably obsolete ... why? -> they are already present!
        return this.translations;
    }
    public int[] pointers(){
        this.get(this.pointers);//probably obsolete
        return this.pointers;
    }
    public int[] idx(){
        this.get(this.tmp_idx);//probably obsolete
        return this.tmp_idx;
    }

    //------------------------------------------
    public double[] value(){this.get(this.tmp_val); return tmp_val; }
    public int[] shape(){this.get(this.tmp_shp); return tmp_shp;}
    public int[] translation(){this.get(this.tmp_tln); return tmp_tln;}
    //------------------------------------------
    private double[] tmp_val;
    private int[] tmp_shp;
    private int[] tmp_tln;
    //-----------------------------
    public @PrivateMemorySpace(8*3) int[] tmp_idx = new int[8*3];
    //-----------------------------
    private int[] mde = {0};
    /**
     *    TENSORS:
     * */
    public double[] values;
    public int[] shapes;
    public int[] translations;

    public int[] pointers;// Pointers f tensors (chronologically)
    /**
     *    tsr pointer++:
     *    +0 -> tsr_ptr: for values
     *    +1 -> tsr_sze: (size) -> negative means: hasGradient()==true
     *       -> full_size: (size) -> 2*tsr_sze if has Gradient()==true
     *
     *    +2 -> shp_ptr: for shapes
     *    +3 -> shp_sze: (size)
     *
     *    +4 -> tln_ptr: for shape index to value translation
     *    +5 -> tln_sze: (size)
     * */
    public TKernel(){
        this.setExplicit(true);
        this.pointers = new int[]{};//0, -initialSize, 0, 0, 0, 0
        this.shapes  = new int[0];
        this.translations = new int[0];
        this.values  = new double[0];
    }
    //-------------------------------------------------------------------
    public int tsr_count(){
        return pointers.length/6;
    }
    //-------------------------------------------------------------------
    // Tensors:
    private int tsr_ptr(int t_id){
        return pointers[t_id*6+0 ];
    }
    private int tsr_sze(int t_id){
        return Math.abs(pointers[t_id*6+1]);
    }
    private int tsr_end(int t_id){
        return (t_id<0)?0:(tsr_ptr(t_id)+tsr_sze(t_id)+ tsr_grd_sze(t_id));
    }

    private void set_tsr_ptr(int t_id, int ptr){
        pointers[t_id*6+0 ] = ptr;
    }
    private void set_tsr_sze(int t_id, int sze){pointers[t_id*6+1]=sze;}

    // Gradients:
    private int tsr_grd_ptr(int t_id){
        return tsr_ptr(t_id)+tsr_sze(t_id);
    }
    private int tsr_grd_sze(int t_id){return tsr_sze(t_id)*((hasGradient(t_id))?1:0);}
    private int tsr_grd_end(int t_id){return tsr_grd_ptr(t_id)+tsr_sze(t_id);}

    // Shapes:
    private int shp_ptr(int t_id){
        return pointers[t_id*6+2 ];
    }
    private int shp_sze(int t_id){
        return pointers[t_id*6+3];
    }
    private int shp_end(int t_id){return shp_ptr(t_id)+shp_sze(t_id);}

    private void set_shp_ptr(int t_id, int ptr){
        pointers[t_id*6+2 ] = ptr;
    }
    private void set_shp_sze(int t_id, int sze){pointers[t_id*6+3]=sze;}

    // Translations:
    private int tln_ptr(int t_id){
        return pointers[t_id*6+4 ];
    }
    private int tln_sze(int t_id){
        return pointers[t_id*6+5];
    }
    private int tln_end(int t_id){return tln_ptr(t_id)+tln_sze(t_id);}

    private void set_tln_ptr(int t_id, int ptr){
        pointers[t_id*6+4 ] = ptr;
    }
    private void set_tln_sze(int t_id, int sze){pointers[t_id*6+5]=sze;}
    //-------------------------------------------------------------------
    private int free_spc(int t_id){
        return  ((t_id+1)*6<this.pointers.length)
                ?(tsr_ptr(t_id+1)-tsr_end(t_id))//=> getting space between t_id and next element
                :this.values.length-tsr_end(t_id);//=> t_id is last element
    }
    //-------------------------------------------------------------------
    private void setNull(int t_id){pointers[t_id*6+1]=0;}
    private boolean ptrIsNull(int t_id){return (t_id<0)?false:(pointers[t_id*6+1]==0);}
    private boolean hasGradient(int t_id){
        return (pointers[t_id*6+1]<0);
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
        this.tmp_shp = shape;
        this.tmp_tln = translation;
        this.put(this.tmp_shp);
        this.put(this.tmp_tln);
        int biggestChunck = 0;
        for(int t_i = -1; t_i<tsr_count(); t_i++){
            biggestChunck =
                (biggestChunck<free_spc(t_i))
                    ?free_spc(t_i)
                    :biggestChunck;
        }
        if(biggestChunck<size){
            int newSpace = (int)(this.values.length* alloc_val_sizer);
            newSpace = (newSpace>size)?newSpace:size;
            this.get(this.values);//TODO: Make a flag so that this is avoided!
            double[] newValues = new double[this.values.length+newSpace];
            for(int i=0; i<values.length; i++){
                newValues[i] = values[i];
            }
            this.values = newValues;
            if(this.pointers.length==0){
                this.pointers = new int[]{0, 0, 0, 0, 0, 0};
            }
            this.put(this.values);
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
                this.put(this.pointers);
                return ptr;
            }
        }
        this.put(this.pointers);
        return 0;//return pointer f alloc_val_sizer
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
            newPointers = new int[pointers.length-6];
            for(int i=0; i<t_id*6; i++){
                newPointers[i] = this.pointers[i];
            }
            for(int i=t_id*6+6; i<this.pointers.length; i++) {
                newPointers[i-6] = this.pointers[i];
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
            newPointers = new int[pointers.length+6];
            for(int i=0; i<t_id*6+6; i++){
                newPointers[i] = this.pointers[i];
            }
            for(int i=t_id*6+6; i<newPointers.length-6; i++){
                newPointers[i+6] = this.pointers[i];
                if(i%6==0){
                    regis[0][mapper[(i/6)]]++;
                }
            }
        }
        this.pointers = newPointers;
        this.put(this.pointers);
        return rgr_ptr;//mapper[t_id];
    }
    /**
     *    returns pointer to shapes/translations elements.
     *    moves this.shapes/this.translations to device if required data is not present.
     *
     * */
    private int alloc_tln(int[] translation){
        int zeros_ptr = 0;
        for(int i=0; i<this.translations.length; i++){
            if(this.translations[i]!=0){
                zeros_ptr = i+1;
            }
            boolean matches = true;
            for(int ii=0; ii<translation.length; ii++){
                if((i+ii>=this.translations.length)||this.translations[i+ii]!=translation[ii]&&this.translations[i+ii]!=0){
                    matches=false;
                }
            }
            if(matches){
                for(int ii=0; ii<translation.length; ii++){
                    this.translations[i+ii]=translation[ii];
                }
                this.put(this.translations);
                return i;
            }
        }
        int[] newTranslations
                = new int[
                this.translations.length+
                        (
                                translation.length>((int)(this.translations.length* alloc_shp_tln_sizer))
                                        ?translation.length
                                        :((int)(this.translations.length* alloc_shp_tln_sizer))
                        )
                ];
        for(int i=0; i<zeros_ptr; i++){
            newTranslations[i] = this.translations[i];
        }
        for(int i=zeros_ptr; i<zeros_ptr+translation.length; i++){
            newTranslations[i] =
                    (i<zeros_ptr+translation.length)
                    ?translation[i-zeros_ptr]
                    :0;
        }
        int ptr = zeros_ptr;
        this.translations = newTranslations;
        this.put(this.translations);
        return ptr;
    }
    private int alloc_shp(int[] shape){
        int zeros_ptr = 0;
        for(int i=0; i<this.shapes.length; i++){
            if(this.shapes[i]!=0){
                zeros_ptr = i+1;
            }
            boolean matches = true;
            for(int ii=0; ii<shape.length; ii++){
                if((i+ii>=this.shapes.length)||this.shapes[i+ii]!=shape[ii]&&this.shapes[i+ii]!=0){
                    matches=false;
                }
            }
            if(matches){
                for(int ii=0; ii<shape.length; ii++){
                    this.shapes[i+ii]=shape[ii];
                }
                this.put(this.shapes);
                return i;
            }
        }
        int[] newShapes =
            new int[
            this.shapes.length+
                (
                    shape.length>((int)(this.shapes.length* alloc_shp_tln_sizer))
                        ?shape.length
                        :((int)(this.shapes.length* alloc_shp_tln_sizer))
                )
            ];
        for(int i=0; i<zeros_ptr; i++){
            newShapes[i] = this.shapes[i];
        }
        for(int i=zeros_ptr; i<zeros_ptr+shape.length; i++){
            newShapes[i] =
                (i<zeros_ptr+shape.length)
                    ?shape[i-zeros_ptr]
                    :0;
        }
        int ptr = zeros_ptr;
        this.shapes = newShapes;
        this.put(this.shapes);
        return ptr;
    }

    /**
     *    Pre-Execution functions (mode setter)
     *    return global size for range creation!
     *    ------------------------------------------------------
     *    ======================================================
    * */
    public int fetch_tsr(int t_id, boolean grd){
        this.tmp_val = new double[tsr_sze(t_id)];
        this.tmp_shp = new int[shp_sze(t_id)];
        this.tmp_tln = new int[tln_sze(t_id)];
        mde = new int[]{(grd)?-4:-3, t_id};
        this.put(this.tmp_val).put(this.tmp_shp).put(this.tmp_tln).put(this.mde);
        int g_sze = tsr_sze(t_id)+shp_sze(t_id)+tln_sze(t_id);
        //System.out.println("fetch: "+g_sze);
        return g_sze;
    }

    public int store_tsr(int t_id, double[] value, boolean grd){
        mde = new int[]{(grd)?-2:-1, t_id};// 1. define if stored as grd or not; 2. specify tsr id;
        this.tmp_val = value;
        this.put(this.mde).put(this.tmp_val);
        int g_sze = tsr_sze(t_id)+shp_sze(t_id)+tln_sze(t_id);
        //System.out.println(g_sze);
        return g_sze;
    }

    public int calculate_tsr(int drn_id, int src1_id, int src2_id, int f_id){

        if(this.mde==null||this.mde.length<3||this.mde[0]!=f_id||this.mde[1]!=drn_id||this.mde[2]!=src1_id||this.mde[3]!=src2_id){
            this.mde = new int[]{f_id, drn_id, src1_id, src2_id};
            this.put(this.mde);
        }
        for(int i=0; i<tmp_idx.length; i++){
            this.tmp_idx[i] = 0;
        }
        this.put(this.tmp_idx);
        return tsr_sze(drn_id);
    }
    public int calculate_tsr(int[] mode){// Mode contains f_id, drain id and source id's !
        if(this.mde==null||this.mde.length<3||this.mde.length!=mode.length){
            this.mde = mode;
            this.put(this.mde);//up
        }
        for(int i=0; i<tmp_idx.length; i++){
            this.tmp_idx[i] = 0;
        }
        this.put(this.tmp_idx);
        return tsr_sze(mode[1]);
    }
    /**
     *    KERNEL RUN:
     *    ==========
     * */
    @Override
    public void run() {
        int gid = this.getGlobalId();
        for(int i=0; i<tmp_idx.length; i++){
            this.tmp_idx[i] = 0;
        }
        if(mde[0]==-5){//cleanup //TODO implement!
        }
        if(mde[0]==-4){//fetch_tsr grad f tensor
            run_tsr_fetch(gid, true);
        }
        if(mde[0]==-3){//fetch_tsr tensor
           run_tsr_fetch(gid, false);
           //this.tmp_val[gid] = gid;
        }
        if(mde[0]==-2){//store_tsr grad f tensor
            run_tsr_store(gid, true);
        }
        if(mde[0]==-1){//store_tsr tensor
            run_tsr_store(gid, false);
        }
        if(mde[0]==0){//Relu
            run_tsr_relu(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==1){//Sigmoid
            run_tsr_sig(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==2){//Tanh
            run_tsr_tnh(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==3){//Quadratic
            run_tsr_qdr(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==4){//Ligmoid
            run_tsr_lig(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==5){//Linear
            run_tsr_lin(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==6){//Gaussian
            run_tsr_gus(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==7){//Absolut
            run_tsr_abs(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==8){//Sinus
            run_tsr_sin(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==9){//Cosinus
            run_tsr_cos(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==10){//Sum
            run_tsr_sum(gid,  mde[1], (mde.length<=3)?-1:mde[mde.length-1]);
        }
        if(mde[0]==11){//Product
            run_tsr_pi(gid,  mde[1], (mde.length<=3)?-1:mde[mde.length-1]);
        }
        if(mde[0]==12){//  ^
            run_tsr_pow(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==13){//  /
            run_tsr_div(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==14){//  *
            run_tsr_mul(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==15){//  %
            run_tsr_mod(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==16){//  -
            run_tsr_sub(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==17){//  +
            run_tsr_add(gid,  mde[1], mde[2], mde[3]);
        }
        if(mde[0]==18){//  tsr_mul
            run_tsr_conv(gid, mde[1], mde[2], mde[3]);
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

    private void run_tsr_fetch(int gid, boolean grd){
        if(gid<tsr_sze(mde[1])){
            this.tmp_val[gid]=this.values[((!grd)?tsr_ptr(mde[1]):tsr_grd_ptr(mde[1]))+gid];
        }else{
            if(gid<(tsr_sze(mde[1])+shp_sze(mde[1]))){
                gid-=tsr_sze(mde[1]);
                this.tmp_shp[gid]=this.shapes[shp_ptr(mde[1])+gid];
            }else{
                if(gid<(tsr_sze(mde[1])+shp_sze(mde[1])+tln_sze(mde[1]))){
                    gid-=(tsr_sze(mde[1])+shp_sze(mde[1]));
                    this.tmp_tln[gid]=this.translations[tln_ptr(mde[1])+gid];
                }
            }
        }
    }

    private void run_tsr_store(int gid, boolean grd){
        if(gid<tsr_sze(mde[1])){
            this.values[((!grd)?tsr_ptr(mde[1]):tsr_grd_ptr(mde[1]))+gid]=this.tmp_val[gid];
        }else{
            if(gid<(tsr_sze(mde[1])+shp_sze(mde[1]))){
                gid-=tsr_sze(mde[1]);
                this.shapes[shp_ptr(mde[1])+gid]=this.tmp_shp[gid];
            }else{
                if(gid<(tsr_sze(mde[1])+shp_sze(mde[1])+tln_sze(mde[1]))){
                    gid-=(tsr_sze(mde[1])+shp_sze(mde[1]));
                    this.translations[tln_ptr(mde[1])+gid]=this.tmp_tln[gid];
                }
            }
        }
    }

    public void run_tsr_relu(int gid, int drn_id, int src_id, int d){
        if (d<0) {
            if (this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)] >= 0) {
                this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                        (this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)] + TFunction.Variables.BIAS) * TFunction.Variables.INCLINATION;
            } else {
                this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                        (this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)] + TFunction.Variables.BIAS) * TFunction.Variables.RELU_INCLINATION;
            }
        } else {
            if (this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)] >= 0) {
                this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                        TFunction.Variables.INCLINATION;
            } else {
                this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                        TFunction.Variables.RELU_INCLINATION;
            }
        }
    }

    public void run_tsr_sig(int gid, int drn_id, int src_id, int d){
        if(d<0){
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    1 / (1 + Math.pow(Math.E, (-this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)])));

        }else{
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                (
                    Math.pow(
                        Math.E,
                        -this.values[tsr_ptr(mde[2+d])+__i_of(gid, mde[2+d], 1)]
                    )
                ) / (Math.pow(
                        (1 + Math.pow(
                                Math.E,
                                -this.values[tsr_ptr(mde[2+d])+__i_of(gid, mde[2+d], 1)]
                        )
                    ), 2)
                        + 2 * Math.pow(
                                Math.E, -this.values[tsr_ptr(mde[2+d])+__i_of(gid, mde[2+d], 1)]));
        }
    }

    public void run_tsr_tnh(int gid, int drn_id, int src_id, int d){
        if(d<0){
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                            / Math.pow(
                                    (1 + Math.pow(
                                            this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                                            , 2)
                                    ), 0.5);

        }else{
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                (1 - Math.pow(
                    (this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                        /
                        Math.pow(
                            (1 + Math.pow(
                                this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                                , 2)
                            ), 0.5
                        )
                    ), 2)
                );

        }
    }
    public void run_tsr_qdr(int gid, int drn_id, int src_id, int d){
        if(d<0){
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.pow(this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)],2);
        }else{
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]*2;
        }
    }
    public void run_tsr_lig(int gid, int drn_id, int src_id, int d){
        if(d<0){
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = (
                this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                + (
                    Math.log(
                        Math.pow(
                            Math.E,
                            -this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]
                        ) + 1
                    ) / Math.log(Math.E)
                )
            );
        }else{
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                1 /
                    (1 + Math.pow(
                            Math.E,
                            this.values[tsr_ptr(mde[2+d])+__i_of(gid, mde[2+d], 1)]
                    )
                );
        }
    }

    public void run_tsr_lin(int gid, int drn_id, int src_id, int d){
        if(d<0){
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)];
        }else{
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    this.values[tsr_ptr(mde[2+d])+__i_of(gid, mde[2+d], 1)];
        }
    }

    public void run_tsr_gus(int gid, int drn_id, int src_id, int d){
        if(d<0){
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    Math.pow(Math.E, -Math.pow(this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)], 2));
        }else{
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    -2 * (this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)])
                            * Math.pow(Math.E, -Math.pow(this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)], 2));
        }
    }

    public void run_tsr_abs(int gid, int drn_id, int src_id, int d){
        this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.abs(this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]);
    }
    public void run_tsr_sin(int gid, int drn_id, int src_id, int d){
        this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.sin(this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]);
    }
    public void run_tsr_cos(int gid, int drn_id, int src_id, int d){
        this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.cos(this.values[tsr_ptr(src_id)+__i_of(gid, src_id, 1)]);
    }

    public void run_tsr_sum(int gid, int drn_id, int d){
        if(d<0){
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = 0;
            for(int i=2; i<(mde.length-1); i++){
                this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] +=
                        this.values[tsr_ptr(mde[i])+__i_of(gid, mde[i], 1)];
            }
        }else{
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    this.values[tsr_ptr(mde[2+d])+__i_of(gid, mde[2+d], 1)];
        }
    }

    public void run_tsr_pi(int gid, int drn_id, int d){
        if(d<0){
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] = 0;
            for(int i=2; i<(mde.length-1); i++){
                this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] *=
                        this.values[tsr_ptr(mde[i])+__i_of(gid, mde[i], 1)];
            }
        }else{
            //TODO: implement
            this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                    this.values[tsr_ptr(mde[2+d])+__i_of(gid, mde[2+d], 1)];//........
        }
    }

    public void run_tsr_pow(int gid, int drn_id, int src1_id, int src2_id){
        this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                Math.pow(
                        this.values[tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)],
                        this.values[tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)]
                        );
    }
    public void run_tsr_div(int gid, int drn_id, int src1_id, int src2_id){
        this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                this.values[tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)]
                        /
                this.values[tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)];
    }
    public void run_tsr_mul(int gid, int drn_id, int src1_id, int src2_id){
        this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                this.values[tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)]
                        *
                this.values[tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)];
    }
    public void run_tsr_mod(int gid, int drn_id, int src1_id, int src2_id){
        this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                ((int)this.values[tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)])
                        %
                ((int)this.values[tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)]);
    }
    public void run_tsr_sub(int gid, int drn_id, int src1_id, int src2_id){
        this.values[tsr_ptr(drn_id)+__i_of(gid, drn_id, 0)] =
                this.values[tsr_ptr(src1_id)+__i_of(gid, src1_id, 1)]
                        -
                this.values[tsr_ptr(src2_id)+__i_of(gid, src2_id, 2)];
    }
    public void run_tsr_add(int gid, int drn_id, int src1_id, int src2_id){
        int i1 = tsr_ptr(drn_id)+__i_of(gid, drn_id, 0);
        int i2 = tsr_ptr(src1_id)+__i_of(gid, src1_id, 1);
        int i3 = tsr_ptr(src2_id)+__i_of(gid, src2_id, 2);
        this.values[i1] =
            this.values[i2]
                +
            this.values[i3];
    }

    public void run_tsr_conv(int gid, int drn_id, int src1_id, int src2_id){
        int ptr_data_src1 = tsr_ptr(src1_id);
        int ptr_data_src2 = tsr_ptr(src2_id);
        int ptr_data_drn = tsr_ptr(drn_id);

        int ptr_shp_src1 = shp_ptr(src1_id);
        int ptr_shp_src2 = shp_ptr(src2_id);
        int ptr_shp_drn  = shp_ptr(drn_id);

        int ptr_tln_src1 = tln_ptr(src1_id);
        int ptr_tln_src2 = tln_ptr(src2_id);
        int ptr_tln_drn  = tln_ptr(drn_id);

        int rank = shp_sze(drn_id);
        int ptr_idx_src1 = 0*rank;
        int ptr_idx_src2 = 1*rank;
        int ptr_idx_drn  = 2*rank;

        int src1End = ptr_shp_src1 + rank;
        int src2End = ptr_shp_src2 + rank;

        //increment on drain:
        for(int i=0; i<gid; i++){//drnSze-1
            __increment_idx(ptr_shp_drn, ptr_idx_drn, rank);
        }
        //increment src accordingly:
        int ri = 0;
        while (ri < rank) {
            if (this.shapes[(ptr_shp_src1+ri)] == this.shapes[(ptr_shp_src2+ri)]) {//setting 0
                this.tmp_idx[(ptr_idx_src1+ri)] = this.tmp_idx[(ptr_idx_drn+ri)];//mtch[mi];
                this.tmp_idx[(ptr_idx_src2+ri)] = this.tmp_idx[(ptr_idx_drn+ri)];//mtch[mi];
            } else if (this.shapes[(ptr_shp_src1+ri)] > this.shapes[(ptr_shp_src2+ri)]) {//setting src1 idx to id idx
                this.tmp_idx[(ptr_idx_src1+ri)] = this.tmp_idx[(ptr_idx_drn+ri)];//mtch[mi];
                this.tmp_idx[(ptr_idx_src2+ri)] = 0;
            } else if (this.shapes[ptr_shp_src1+ri] < this.shapes[(ptr_shp_src2+ri)]) {//setting src2 idx to id idx
                this.tmp_idx[(ptr_idx_src1+ri)] = 0;
                this.tmp_idx[(ptr_idx_src2+ri)] = this.tmp_idx[(ptr_idx_drn+ri)];//mtch[mi];
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
                int i1 = __i_of_idx_tln(ptr_tln_src1, ptr_idx_src1, rank); //(int ptr_tln, int[] tln, int[] idx, int rank)
                int i2 = __i_of_idx_tln(ptr_tln_src2, ptr_idx_src2, rank);
                int gogo = (ptr_data_src1 + i1);
                int gugu = (ptr_data_src2 + i2);
                value +=
                     this.values[(ptr_data_src1 + i1)]
                         *
                     this.values[(ptr_data_src2 + i2)];
                incrementing = true;
                ri=0;
            } else {//incrementing:
                if (this.tmp_idx[(ptr_idx_src1+ri)] < this.shapes[(ptr_shp_src1+ri)] && this.tmp_idx[(ptr_idx_src2+ri)] < this.shapes[(ptr_shp_src2+ri)]) {
                    this.tmp_idx[(ptr_idx_src1+ri)]++;
                    this.tmp_idx[(ptr_idx_src2+ri)]++;
                    if (this.tmp_idx[(ptr_idx_src1+ri)] == this.shapes[(ptr_shp_src1+ri)] || this.tmp_idx[(ptr_idx_src2+ri)] == this.shapes[(ptr_shp_src2+ri)]) {
                        if (((ptr_shp_src1+ri) == (src1End - 1) || (ptr_shp_src2+ri) == (src2End - 1))) {
                            running = false;
                        }
                        if (this.shapes[(ptr_shp_src1+ri)] == this.shapes[(ptr_shp_src2+ri)]) {//setting 0
                            this.tmp_idx[(ptr_idx_src1+ri)] = this.tmp_idx[(ptr_idx_drn+ri)];//mtch[mi];
                            this.tmp_idx[(ptr_idx_src2+ri)] = this.tmp_idx[(ptr_idx_drn+ri)];//mtch[mi];
                        } else if (this.shapes[(ptr_shp_src1+ri)] > this.shapes[(ptr_shp_src2+ri)]) {//setting hdr1 idx to id idx
                            this.tmp_idx[(ptr_idx_src1+ri)] = this.tmp_idx[(ptr_idx_drn+ri)];//mtch[mi];
                            this.tmp_idx[(ptr_idx_src2+ri)] = 0;
                        } else if (this.shapes[(ptr_shp_src1+ri)] < this.shapes[(ptr_shp_src2+ri)]) {//setting hdr2 idx to id idx
                            this.tmp_idx[(ptr_idx_src1+ri)] = 0;
                            this.tmp_idx[(ptr_idx_src2+ri)] = this.tmp_idx[(ptr_idx_drn+ri)];//mtch[mi];
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
        int i = __i_of_idx_tln(ptr_tln_drn, ptr_idx_drn, rank);
        this.values[(ptr_data_drn + i)] = value;
    }
    //Helper methods for tsr conv:
    private int __increment_At(int ri, int idx_ptr, int shp_ptr) {
        if (this.tmp_idx[idx_ptr+ri] < (this.shapes[shp_ptr+ri])) {//fixed
            this.tmp_idx[idx_ptr+ri]++;
            if (this.tmp_idx[idx_ptr+ri] == (this.shapes[shp_ptr+ri])) {
                this.tmp_idx[idx_ptr+ri] = 0;
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
    private int __i_of_idx_tln(int ptr_tln, int idx_ptr, int rank) {
        int i = 0;
        for (int ii = 0; ii < rank; ii++) {
            i += this.translations[ptr_tln+ii] * this.tmp_idx[idx_ptr+ii];
        }
        return i;
    }

    private int __i_of(int gid, int t_id, int num){
        int ptr_shp  = shp_ptr(t_id);
        int ptr_tln  = tln_ptr(t_id);
        int rank     = shp_sze(t_id);
        int ptr_idx  = rank*num;
        for(int i=0; i<gid; i++){
            __increment_idx(ptr_shp, ptr_idx, rank);
        }
        return __i_of_idx_tln(ptr_tln, ptr_idx, rank);
    }

}
