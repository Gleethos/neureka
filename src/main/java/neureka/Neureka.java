/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

     _   _                     _
    | \ | |                   | |
    |  \| | ___ _   _ _ __ ___| | ____ _
    | . ` |/ _ \ | | | '__/ _ \ |/ / _` |
    | |\  |  __/ |_| | | |  __/   < (_| |
    |_| \_|\___|\__,_|_|  \___|_|\_\__,_|

    This is a central singleton class used to configure the Neureka library.

*/

package neureka;


import neureka.backend.api.Operation;
import neureka.backend.api.OperationContext;
import neureka.devices.opencl.CLContext;
import neureka.dtype.custom.F64;
import neureka.utility.Messages;
import neureka.utility.SettingsLoader;
import neureka.utility.TsrAsString;
import org.slf4j.Logger;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.ServiceLoader;

/**
 *    {@link Neureka} is the key access point for thread local / global library settings ( see{@link Settings})
 *    as well as execution contexts (see {@link OperationContext})
 *    and pre-instantiated {@link neureka.calculus.Function}s.
 *    {@link Neureka} exposes the execution context via the {@link #context()} method,
 *    the library settings which govern the behaviour of various library components
 *    can be accessed via the {@link #settings()} method.
 *    Common functions can be accessed within a given {@link OperationContext} instance based on which they were built.
 *    If one wishes to modify the default library settings it is possible to do so by editing
 *    the "library_settings.groovy" DSL file.
 */
public final class Neureka
{
    private static final ThreadLocal<Neureka> _INSTANCES;
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(Neureka.class);

    /**
     *  The current semantic version of this library build.
     */
    private static String _VERSION = "0.7.0";

    /**
     *  The truth value determining if OpenCL is available or not.
     */
    private static final boolean _OPENCL_AVAILABLE;

    static
    {
        _INSTANCES = new ThreadLocal<>();
        _OPENCL_AVAILABLE = Utility.isPresent( "org.jocl.CL" );
    }

    private final Settings _settings;
    private final Utility _utility;

    /**
     *  This is a lazy reference to the so called {@link OperationContext}
     *  which will instantiated and populated as soon as the {@link #context()}
     *  method is being called for the first time.
     *  This context contains anything needed to perform operations
     *  on tensors on using different {@link neureka.calculus.Function}
     *  or {@link neureka.devices.Device} implementation instances.
     */
    private OperationContext _context;

    public OperationContext context() {
        if ( _context == null ) {
            _context = new OperationContext();
            // loading operations!
            ServiceLoader<Operation> serviceLoader = ServiceLoader.load( Operation.class );

            //checking if load was successful
            for ( Operation operation : serviceLoader ) {
                assert operation.getFunction() != null;
                assert operation.getOperator() != null;
                if ( operation.getFunction() == null ) log.error(Messages.Operations.illegalStateFor( "function" ) );
                if ( operation.getOperator() == null ) log.error(Messages.Operations.illegalStateFor( "operator" ) );
                _context.addOperation(operation);
                log.debug( Messages.Operations.loaded(operation) );
            }
            if ( _OPENCL_AVAILABLE )
                _context.set( new CLContext() );
            else
                log.warn( Messages.OpenCL.clContextCreationFailed() );
        }
        return _context;
    }

    private Neureka() {
        _settings = new Settings();
        _utility = new Utility();
    }

    /**
     *  The {@link Neureka} class represents the configuration of this library.
     *  Instances of this configuration are stored local to every thread in order to make
     *  both the library settings as well as the execution context threadsafe!
     *  This method will return the {@link Neureka} instance which corresponds to the thread calling it.
     *
     * @return The thread local library configuration state called {@link Neureka}.
     */
    public static Neureka get() {
        Neureka n = _INSTANCES.get();
        if ( n == null ) {
            n = new Neureka();
            synchronized ( Neureka.class ) {
                set( n );
                n.reset(); // Initial reset must be synchronized because of dependency issues!
            }
        }
        return n;
    }

    /**
     *  {@link Neureka} is a thread local singleton.
     *  Therefore, this method will only set the provided {@link Neureka} instance
     *  for the thread which is calling this method.
     *  Other threads calling the {@link #get()} method to retrieve the instance
     *  will get their own instance...
     *  (This can theoretically be bypassed by sharing instances)
     *
     * @param instance The {@link Neureka} instance which ought to be set as thread local singleton.
     */
    public static void set( Neureka instance ) {
        _INSTANCES.set(instance);
    }

    public static Neureka configure( Object closure ) {
        Object o = SettingsLoader.tryGroovyClosureOn(closure, Neureka.get());
        if ( o instanceof String ) _VERSION = (String) o;
        return Neureka.get();
    }

    /**
     * @return The truth value determining if OpenCL is accessible.
     */
    public boolean canAccessOpenCL() {
        return _OPENCL_AVAILABLE;
    }

    /**
     * @return An instance of library wide {@link Settings} determining the behaviour of many classes...
     */
    public Settings settings() {
        return _settings;
    }

    public Settings settings(Object closure) {
        SettingsLoader.tryGroovyClosureOn(closure, _settings);
        return _settings;
    }

    /**
     * @return An instance of an utility class useful for loading resources or checking if they are even available.
     */
    public Utility utility() {
        return _utility;
    }

    /**
     * @return The semantic version of the Neureka library.
     */
    public static String version() {
        return _VERSION;
    }

    /**
     *  This method will try to reload the "library_settings.groovy" script
     *  which will re-configure the library wide {@link Settings} instance nested inside {@link Neureka}.
     *  If the execution of this file fails then the settings will be reverted to a hardcoded default state.
     */
    public void reset() {
        try {
            SettingsLoader.loadProperties(this);
            // The following can be used when one desires a Groovy DSL as settings source!:
            //SettingsLoader.tryGroovyScriptsOn(this, script -> new GroovyShell(getClass().getClassLoader()).evaluate(script));
        } catch ( Exception e ) {
            settings().autograd().setIsRetainingPendingErrorForJITProp( true );
            settings().autograd().setIsApplyingGradientWhenTensorIsUsed( true );
            settings().autograd().setIsApplyingGradientWhenRequested( true );
            settings().indexing().setIsUsingArrayBasedIndexing( true );
            settings().debug().setIsKeepingDerivativeTargetPayloads( false );
            settings().view().setIsUsingLegacyView( false );
        }
    }

    private boolean _currentThreadIsNotAuthorized() {
        return !this.equals(_INSTANCES.get());
    }

    public String toString() {
        return "Neureka(" +
                    "settings=" + this._settings + ", " +
                    "utility=" + this._utility + ", " +
                    "context=" + this._context +
                ")";
    }

    public OperationContext getContext() {
        return this._context;
    }

    public void setContext(OperationContext _context) {
        this._context = _context;
    }

    /**
     *  This class hosts the settings of the {@link Neureka} instance which will be used throughout the library.
     */
    public class Settings
    {
        private final Debug    _debug;
        private final AutoGrad _autograd;
        private final Indexing _indexing;
        private final View     _view;
        private final NDim     _ndim;
        private final DType    _dtype;

        private boolean _isLocked = false;

        private Settings() {
            _debug    = new Debug();
            _autograd = new AutoGrad();
            _indexing = new Indexing();
            _view     = new View();
            _ndim     = new NDim();
            _dtype    = new DType();
        }

        public Debug debug() { return _debug; }

        public Debug debug(Object closure) {
            SettingsLoader.tryGroovyClosureOn(closure, _debug);
            return _debug;
        }

        public AutoGrad autograd() {
            return _autograd;
        }

        public AutoGrad autograd( Object closure ) {
            SettingsLoader.tryGroovyClosureOn( closure, _autograd );
            return _autograd;
        }

        public Indexing indexing() {
            return _indexing;
        }

        public Indexing indexing( Object closure ) {
            SettingsLoader.tryGroovyClosureOn( closure, _indexing );
            return _indexing;
        }

        public View view() {
            return _view;
        }

        public View view( Object closure ) {
            SettingsLoader.tryGroovyClosureOn( closure, _view );
            return _view;
        }

        public NDim ndim() {
            return _ndim;
        }

        public NDim ndim( Object closure ) {
            SettingsLoader.tryGroovyClosureOn( closure, _ndim );
            return _ndim;
        }

        public DType dtype() {
            return _dtype;
        }

        public DType dtype( Object closure ) {
            SettingsLoader.tryGroovyClosureOn( closure, _dtype );
            return _dtype;
        }

        public boolean isLocked() {
            return  _isLocked;
        }

        public void setIsLocked(boolean locked) {
            _isLocked = locked;
        }

        public String toString() {
            return "Neureka.Settings(" +
                        "debug=" + this._debug + ", " +
                        "autograd=" + this._autograd + ", " +
                        "indexing=" + this._indexing + ", " +
                        "view=" + this._view + ", " +
                        "ndim=" + this._ndim + ", " +
                        "dtype=" + this._dtype + ", " +
                        "isLocked=" + this.isLocked() +
                    ")";
        }

        
        public class Debug
        {
            private boolean _isKeepingDerivativeTargetPayloads = false;

            /**
             * Every derivative is calculated with respect to some graph node.
             * Graph nodes contain payload tensors.
             * A tensor might not always be used for backpropagation.
             * Therefore it will be deleted if possible.
             * Targeted tensors are either leave tensors (They require gradients)
             * or they are angle points between forward- and reverse-mode-AutoDiff!
             * In this case:
             * If the tensor is not needed for backpropagation it will be deleted.
             * The graph node will dereference the tensor either way.
             *
             * The flag determines this behavior with respect to target nodes.
             * It is used in the test suit to validate that the right tensors were calculated.
             * This flag should not be modified in production! (memory leak)
             */
            public boolean isKeepingDerivativeTargetPayloads() {
                return _isKeepingDerivativeTargetPayloads;
            }

            /**
             * Every derivative is calculated with respect to some graph node.
             * Graph nodes contain payload tensors.
             * A tensor might not always be used for backpropagation.
             * Therefore it will be deleted if possible.
             * Targeted tensors are either leave tensors (They require gradients)
             * or they are angle points between forward- and reverse-mode-AutoDiff!
             * In this case:
             * If the tensor is not needed for backpropagation it will be deleted.
             * The graph node will dereference the tensor either way.
             *
             * The flag determines this behavior with respect to target nodes.
             * It is used in the test suit to validate that the right tensors were calculated.
             * This flag should not be modified in production! (memory leak)
             */
            public void setIsKeepingDerivativeTargetPayloads(boolean keep) {
                if ( _isLocked || _currentThreadIsNotAuthorized()) return;
                _isKeepingDerivativeTargetPayloads = keep;
            }

            public String toString() {
                return "Neureka.Settings.Debug(" +
                            "isKeepingDerivativeTargetPayloads=" + this.isKeepingDerivativeTargetPayloads() +
                        ")";
            }
        }

        
        public class AutoGrad // Auto-Grad/Differentiation
        {
            private boolean _isPreventingInlineOperations = true;
            private boolean _isRetainingPendingErrorForJITProp = true;
            private boolean _isApplyingGradientWhenTensorIsUsed = true;
            private boolean _isApplyingGradientWhenRequested = true;

            /**
             *  Inline operations are operations where the data of a tensor passed into an operation
             *  is being modified.
             *  Usually the result of an operation is stored inside a new tensor.
             */
            public boolean isPreventingInlineOperations() {
                return _isPreventingInlineOperations;
            }

            /**
             *  Inline operations are operations where the data of a tensor passed into an operation
             *  is being modified.
             *  Usually the result of an operation is stored inside a new tensor.
             */
            public void setIsPreventingInlineOperations(boolean prevent) {
                if ( _isLocked || _currentThreadIsNotAuthorized()) return;
                _isPreventingInlineOperations = prevent;
            }

            /**
             *  This flag enables an optimization technique which only propagates error values to
             *  gradients if needed by a tensor (the tensor is used again) and otherwise accumulate them
             *  at divergent differentiation paths within the computation graph.<br>
             *  If the flag is set to true <br>
             *  then error values will accumulate at such junction nodes.
             *  This technique however uses more memory but will
             *  improve performance for some networks substantially.
             *  The technique is termed JIT-Propagation.
             */
            public boolean isRetainingPendingErrorForJITProp() {
                return _isRetainingPendingErrorForJITProp;
            }

            /**
             *  This flag enables an optimization technique which only propagates error values to
             *  gradients if needed by a tensor (the tensor is used again) and otherwise accumulate them
             *  at divergent differentiation paths within the computation graph.<br>
             *  If the flag is set to true <br>
             *  then error values will accumulate at such junction nodes.
             *  This technique however uses more memory but will
             *  improve performance for some networks substantially.
             *  The technique is termed JIT-Propagation.
             */
            public void setIsRetainingPendingErrorForJITProp(boolean retain) {
                if ( _isLocked || _currentThreadIsNotAuthorized()) return;
                _isRetainingPendingErrorForJITProp = retain;
            }

            /**
             *  Gradients will automatically be applied (or JITed) to tensors as soon as
             *  they are being used for calculation ({@link neureka.autograd.GraphNode} instantiation).
             *  This feature works well with JIT-Propagation.
             */
            public boolean isApplyingGradientWhenTensorIsUsed() {
                return _isApplyingGradientWhenTensorIsUsed;
            }

            /**
             *  Gradients will automatically be applied (or JITed) to tensors as soon as
             *  they are being used for calculation ({@link neureka.autograd.GraphNode} instantiation).
             *  This feature works well with JIT-Propagation.
             */
            public void setIsApplyingGradientWhenTensorIsUsed(boolean apply) {
                if ( _isLocked || _currentThreadIsNotAuthorized()) return;
                _isApplyingGradientWhenTensorIsUsed = apply;
            }

            /**
             *  Gradients will only be applied if requested.
             *  Usually this happens immediately, however
             *  if the flag <i>'applyGradientWhenTensorIsUsed'</i> is set
             *  to true, then the tensor will only be updated by its
             *  gradient if requested AND the tensor is used for calculation!
             *  ({@link neureka.autograd.GraphNode} instantiation).  <br> <br>
             *
             *  This flag works alongside two other autograd features which can be enables by flipping the feature flags <br>
             *  <i>'isApplyingGradientWhenRequested'</i> and <i>'isApplyingGradientWhenTensorIsUsed'</i><br>
             *  As the first flag name suggests gradients will be applied to their tensors when it is set to true,
             *  however this will only happened when the second flag is set to true as well, because otherwise gradients
             *  wouldn't be applied to their tensors automatically in the first place... <br>
             *  <br>
             *  Setting both flags to true will inhibit the effect of the second setting <i>'isApplyingGradientWhenTensorIsUsed'</i>
             *  unless a form of "permission" is being signaled to the autograd system.
             *  This signal comes in the form of a "request" flag which marks a tensor as <b>allowed to
             *  be updated by its gradient</b>. This request can be dispatched to a {@link Tsr}
             *  by setting {@link Tsr#setGradientApplyRequested(boolean)} to {@code true}.<br>
             */
            public boolean isApplyingGradientWhenRequested() {
                return _isApplyingGradientWhenRequested;
            }

            /**
             * Gradients will only be applied if requested.
             * Usually this happens immediately, however
             * if the flag <i>'applyGradientWhenTensorIsUsed'</i> is set
             * to true, then the tensor will only be updated by its
             * gradient if requested AND the tensor is used for calculation!
             * ({@link neureka.autograd.GraphNode} instantiation).
             */
            public void setIsApplyingGradientWhenRequested(boolean apply) {
                if ( _isLocked || _currentThreadIsNotAuthorized()) return;
                _isApplyingGradientWhenRequested = apply;
            }

            public String toString() {
                return "Neureka.Settings.AutoGrad(" +
                            "isPreventingInlineOperations=" + this.isPreventingInlineOperations() + ", " +
                            "isRetainingPendingErrorForJITProp=" + this.isRetainingPendingErrorForJITProp() + ", " +
                            "isApplyingGradientWhenTensorIsUsed=" + this.isApplyingGradientWhenTensorIsUsed() + ", " +
                            "isApplyingGradientWhenRequested=" + this.isApplyingGradientWhenRequested() +
                        ")";
            }
        }

        
        public class Indexing
        {
            private boolean _isUsingArrayBasedIndexing = true;

            public boolean isUsingArrayBasedIndexing() {
                return _isUsingArrayBasedIndexing;
            }

            public void setIsUsingArrayBasedIndexing( boolean thorough ) {
                if ( _isLocked || _currentThreadIsNotAuthorized()) return;
                _isUsingArrayBasedIndexing = thorough;
            }

            public String toString() {
                return "Neureka.Settings.Indexing(" +
                            "isUsingArrayBasedIndexing=" + this.isUsingArrayBasedIndexing() +
                        ")";
            }
        }

        
        public class View
        {
            View(){
                _asString = new HashMap<>();
                _asString.put( TsrAsString.Should.BE_SHORTENED_BY,      50    );
                _asString.put( TsrAsString.Should.BE_COMPACT,           true  );
                _asString.put( TsrAsString.Should.BE_FORMATTED,         true  );
                _asString.put( TsrAsString.Should.HAVE_GRADIENT,        true  );
                _asString.put( TsrAsString.Should.HAVE_PADDING_OF,      6     );
                _asString.put( TsrAsString.Should.HAVE_VALUE,           true  );
                _asString.put( TsrAsString.Should.HAVE_RECURSIVE_GRAPH, false );
                _asString.put( TsrAsString.Should.HAVE_DERIVATIVES,     false );
                _asString.put( TsrAsString.Should.HAVE_SHAPE,           true  );
            }

            private boolean _isUsingLegacyView = false;

            private Map<TsrAsString.Should, Object> _asString;


            public boolean isUsingLegacyView() {
                return _isUsingLegacyView;
            }

            public void setIsUsingLegacyView(boolean enabled) {
                if ( _isLocked || _currentThreadIsNotAuthorized()) return;
                _isUsingLegacyView = enabled;
            }

            public Map<TsrAsString.Should, Object> getAsString() {
                return _asString;
            }

            public void setAsString( Map<TsrAsString.Should, Object> should ) {
                _asString = should;
            }

            public void setAsString( String modes ) {
                setAsString( TsrAsString.Util.configFromCode( modes ) );
            }

            public String toString() {
                return "Neureka.Settings.View(" +
                            "isUsingLegacyView=" + this.isUsingLegacyView() + ", " +
                            "asString=" + this.getAsString() +
                        ")";
            }
        }

        
        public class NDim
        {
            /**
             *  The {@link neureka.ndim.config.types.complex.ComplexDefaultNDConfiguration}
             *  class stores shape, translation... as cached int arrays.
             *  Disabling this flag allows for custom 1D, 2D, 3D classes to be loaded. (Improves memory locality)
             */
            private boolean _isOnlyUsingDefaultNDConfiguration = false;

            public boolean isOnlyUsingDefaultNDConfiguration() {
                return _isOnlyUsingDefaultNDConfiguration;
            }

            public void setIsOnlyUsingDefaultNDConfiguration(boolean enabled) {
                if ( _isLocked || _currentThreadIsNotAuthorized()) return;
                _isOnlyUsingDefaultNDConfiguration = enabled;
            }

            public String toString() {
                return "Neureka.Settings.NDim(" +
                              "isOnlyUsingDefaultNDConfiguration=" + this.isOnlyUsingDefaultNDConfiguration() +
                        ")";
            }
        }

        
        public class DType {

            private Class<?> _defaultDataTypeClass = F64.class;

            private boolean _isAutoConvertingExternalDataToJVMTypes = true;

            public Class<?> getDefaultDataTypeClass() {
                return _defaultDataTypeClass;
            }

            public void setDefaultDataTypeClass( Class<?> dtype ) {
                if ( _isLocked || _currentThreadIsNotAuthorized()) return;
                _defaultDataTypeClass = dtype;
            }

            public boolean getIsAutoConvertingExternalDataToJVMTypes() {
                return _isAutoConvertingExternalDataToJVMTypes;
            }

            public void setIsAutoConvertingExternalDataToJVMTypes( boolean autoConvert ) {
                if ( _isLocked || _currentThreadIsNotAuthorized()) return;
                _isAutoConvertingExternalDataToJVMTypes = autoConvert;
            }

            public String toString() {
                return "Neureka.Settings.DType(_defaultDataTypeClass=" + this.getDefaultDataTypeClass() + ", _isAutoConvertingExternalDataToJVMTypes=" + this._isAutoConvertingExternalDataToJVMTypes + ")";
            }
        }

    }

    public static class Utility
    {
        /**
         * Helper method which reads the file with the given name and returns
         * the contents of this file as a String. Will exit the application
         * if the file can not be read.
         *
         * @param path The path to the jar resource.
         * @return The contents of the file
         */
        public String readResource( String path ) {
            InputStream stream = getClass().getClassLoader().getResourceAsStream( path );
            try {
                BufferedReader br = new BufferedReader(new InputStreamReader( stream ));
                StringBuilder sb = new StringBuilder();
                String line = "";
                while ( line != null ) {
                    line = br.readLine();
                    if ( line != null ) sb.append( line ).append( "\n" );
                }
                return sb.toString();
            } catch ( IOException e ) {
                e.printStackTrace();
                System.exit( 1 );
                return null;
            }
        }

        /**
         *  This method checks if a class addressed by it's name has been loaded into the runtime.
         *
         * @param className The class whose presents ought to be checked.
         * @return The truth value determining if the class is present or not.
         */
        public static boolean isPresent( String className ) {
            boolean found = false;
            String groovyInfo = ( (className.toLowerCase().contains("groovy") ) ? " Neureka settings uninitialized!" : "" );
            String cause = " unknown ";
            try {
                Class.forName( className );
                found = true;
            } catch ( Throwable ex ) {// Class or one of its dependencies is not present...
                cause = ex.getMessage();
            } finally {
                if ( !found ) {
                    System.out.println(
                            "[Info]: '"+className+"' dependencies not found!"+groovyInfo+"\n" +
                            "[Cause]: "+cause+"\n" +
                             findTip().bootstrapTip()
                    );
                }
                return found;
            }
        }

        public static Messages.OpenCL.Tips findTip() {
            /*
                // Check lib: $ ls -l /usr/lib/libOpenCL*

                       UBUNTU:
                       $ sudo apt update
                       $ sudo apt install ocl-icd-opencl-dev
                       // Now libOpenCL.so should be located at /usr/lib/x86_64-linux-gnu/libOpenCL.so
                       $ sudo ubuntu-drivers autoinstall
                       ///////////////////
                       Switch ing from nvidia to amd:
                       The following command will remove the proprietary Nvidia driver:

                           $ sudo dpkg -P $(dpkg -l | grep nvidia-driver | awk '{print $2}')
                           $ sudo apt autoremove

                       Switch back to nouveau driver:

                           $ sudo apt install xserver-xorg-video-nouveau


                       FEDORA:
                       $ sudo dnf install ocl-icd-devel

                */
            Properties properties = new Properties();
            String osName = System.getProperty("os.name");
            if ( osName.toLowerCase().contains("linux") ) {
                String[] cmd = {"/bin/sh", "-c", "cat /etc/*-release"};
                try {
                    Process p = Runtime.getRuntime().exec(cmd);
                    BufferedReader bri = new BufferedReader(new InputStreamReader(p.getInputStream()));
                    StringBuilder text = new StringBuilder();
                    String line = "";
                    while ((line = bri.readLine()) != null) {
                        text.append(line);
                        text.append("\n");
                    }
                    properties.load(new StringReader(text.toString()));

                } catch (IOException e) {

                    e.printStackTrace();
                }
            }
            else // We just assume windows for now.
            {
                try {
                    properties.load(new StringReader("NAME=\"Windows\""));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            //---
            String foundOS = properties.getProperty("NAME").toLowerCase().replace("\"", "");
            switch ( foundOS ) {
                case "ubuntu": return Messages.OpenCL.Tips.UBUNTU;
                case "fedora": return Messages.OpenCL.Tips.FEDORA;
                case "windows": return Messages.OpenCL.Tips.WINDOWS;
            }
            return Messages.OpenCL.Tips.UNKNOWN;
        }

    }

}
