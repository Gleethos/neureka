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

    _____      _   _   _                 _                     _
   / ____|    | | | | (_)               | |                   | |
  | (___   ___| |_| |_ _ _ __   __ _ ___| |     ___   __ _  __| | ___ _ __
   \___ \ / _ \ __| __| | '_ \ / _` / __| |    / _ \ / _` |/ _` |/ _ \ '__|
   ____) |  __/ |_| |_| | | | | (_| \__ \ |___| (_) | (_| | (_| |  __/ |
  |_____/ \___|\__|\__|_|_| |_|\__, |___/______\___/ \__,_|\__,_|\___|_|
                                __/ |
                               |___/

    A simply utility class used by the Neureka singleton instance for configuration loading...
*/

package neureka.common.utility;

import neureka.Neureka;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Properties;
import java.util.function.Consumer;

/**
 *  This class is a helper class for {@link Neureka} instances (Thread local singletons).
 *  It loads the settings property file and interprets its contents
 *  which are then translated to the {@link neureka.Neureka.Settings}.
 */
public class SettingsLoader
{
    private static final Logger _LOG = LoggerFactory.getLogger(SettingsLoader.class);
    private static String _settings_source;
    private static String _setup_source;

    public static void loadProperties( Neureka instance ) {
        try (
                final InputStream stream = instance.getClass()
                                                    .getClassLoader()
                                                    .getResourceAsStream( "library_settings.properties" )
        ) {
            Properties properties = new Properties();
            properties.load( stream );
            Neureka.Settings s = instance.settings();
            new TypeChecker( properties )
                    .checkAndAssign("debug.isKeepingDerivativeTargetPayloads"      , Boolean.class, v -> s.debug().setIsKeepingDerivativeTargetPayloads(v)                     )//~= false
                    .checkAndAssign("autograd.isPreventingInlineOperations"        , Boolean.class, v -> s.autograd().setIsPreventingInlineOperations(v)                       )//~= true
                    .checkAndAssign("autograd.isRetainingPendingErrorForJITProp"   , Boolean.class, v -> s.autograd().setIsRetainingPendingErrorForJITProp(v)                  )//~= true
                    .checkAndAssign("autograd.isApplyingGradientWhenTensorIsUsed"  , Boolean.class, v -> s.autograd().setIsApplyingGradientWhenTensorIsUsed(v)                 )//~= true
                    .checkAndAssign("autograd.isApplyingGradientWhenRequested"     , Boolean.class, v -> s.autograd().setIsApplyingGradientWhenRequested(v)                    )//~= true
                    .checkAndAssign("indexing.isUsingArrayBasedIndexing"           , Boolean.class, v -> s.indexing().setIsUsingArrayBasedIndexing(v)                               )//~= false
                    .checkAndAssign("view.tensors.HAVE_ROW_LIMIT_OF"    , Integer.class, v -> s.view().getTensorSettings().setRowLimit(v)                              )//~= 50,
                    .checkAndAssign("view.tensors.BE_COMPACT"           , Boolean.class, v -> s.view().getTensorSettings().setIsScientific(v)                              )//~= true,
                    .checkAndAssign("view.tensors.BE_FORMATTED"         , Boolean.class, v -> s.view().getTensorSettings().setIsMultiline(v)                        )//~= true,
                    .checkAndAssign("view.tensors.HAVE_GRADIENT"        , Boolean.class, v -> s.view().getTensorSettings().setHasGradient(v)                        )//~= true,
                    .checkAndAssign("view.tensors.HAVE_SLIM_NUMBERS"    , Boolean.class, v -> s.view().getTensorSettings().setHasSlimNumbers(v)              )//~= false,
                    .checkAndAssign("view.tensors.HAVE_PADDING_OF"      , Integer.class, v -> s.view().getTensorSettings().setCellSize(v)                             )//~= 6,
                    .checkAndAssign("view.tensors.HAVE_VALUE"           , Boolean.class, v -> s.view().getTensorSettings().setHasValue(v)                           )//~= true,
                    .checkAndAssign("view.tensors.HAVE_RECURSIVE_GRAPH" , Boolean.class, v -> s.view().getTensorSettings().setHasRecursiveGraph(v)                  )//~= false,
                    .checkAndAssign("view.tensors.HAVE_DERIVATIVES"     , Boolean.class, v -> s.view().getTensorSettings().setHasDerivatives(v)                           )//~= false,
                    .checkAndAssign("view.tensors.HAVE_SHAPE"           , Boolean.class, v -> s.view().getTensorSettings().setHasShape(v)                                 )//~= true,
                    .checkAndAssign("view.tensors.BE_CELL_BOUND"        , Boolean.class, v -> s.view().getTensorSettings().setIsCellBound(v)                              )//~= false
                    .checkAndAssign("view.tensors.HAVE_POSTFIX"         , String.class,  v -> s.view().getTensorSettings().setPostfix(v)                                  )//~= ""
                    .checkAndAssign("view.tensors.HAVE_PREFIX"          , String.class,  v -> s.view().getTensorSettings().setPrefix(v)                                  )//~= ""
                    .checkAndAssign("view.tensors.indent"               , String.class,  v -> s.view().getTensorSettings().setIndent(v)                                 )//~= ""
                    .checkAndAssign("view.tensors.LEGACY"               , Boolean.class, v -> s.view().getTensorSettings().setIsLegacy(v)                                 )//~= ""
                    .checkAndAssign("ndim.isOnlyUsingDefaultNDConfiguration"       , Boolean.class, v -> s.ndim().setIsOnlyUsingDefaultNDConfiguration(v)                      )//~= false
                    .checkAndAssign("dtype.defaultDataTypeClass"                   , Class.class,   v -> s.dtype().setDefaultDataTypeClass(v)                                  )
                    .checkAndAssign("dtype.isAutoConvertingExternalDataToJVMTypes" , Boolean.class, v -> s.dtype().setIsAutoConvertingExternalDataToJVMTypes(v)                );

        } catch ( IOException e ) {
            _LOG.error("Failed to load library settings!", e);
        }
    }

    private static class TypeChecker {

        private final Properties _properties;

        TypeChecker( Properties properties ) { _properties = properties; }

        public <T> TypeChecker checkAndAssign( String key, Class<T> typeClass, Consumer<T> assignment ) {
            Object value = _properties.get( key );
            if ( value == null || value.getClass() != String.class ) {
                _LOG.warn("Illegal value '"+value+"' found for property name '"+key+"' in library settings.");
                return this;
            }
            String asString = (String) value;
            T toBeAssigned = null;
            if ( typeClass == Class.class ) {
                try {
                    try { getClass().getClassLoader().loadClass("neureka.dtype.custom."+asString); } catch (Exception e) {}
                    toBeAssigned = (T) Class.forName("neureka.dtype.custom."+asString);
                }
                catch ( ClassNotFoundException e ) {
                    _LOG.warn("Failed to find class '"+asString+"' for property name '"+key+"'.");
                    return this;
                }
            }
            else if ( typeClass == Boolean.class ) {
                try { toBeAssigned = (T) Boolean.valueOf(Boolean.parseBoolean(asString)); }
                catch ( Exception e ) {
                    _LOG.warn("Failed to parse boolean from value '"+asString+"' for property name '"+key+"'.");
                    return this;
                }
            }
            else if ( typeClass == Integer.class ) {
                try { toBeAssigned = (T) Integer.valueOf(Integer.parseInt(asString)); }
                catch ( Exception e ) {
                    _LOG.warn("Failed to parse integer from value '"+asString+"' for property name '"+key+"'.");
                    return this;
                }
            }
            else if ( typeClass == String.class ) {
                if ( asString.matches("^\"(.*)\"$|^'(.*)'$") ) // Quotes will be trimmed!
                    asString = asString.substring(1, asString.length()-1);
                toBeAssigned = (T) asString;
            }
            assignment.accept(toBeAssigned);
            return this;
        }
    }

    /**
     *  This method makes it possible to configure the library via a Groovy DSL!
     *
     * @param closure A Groovy closure which should be called with the provided delegate.
     * @param delegate The delegate for the provided closure (Can be a settings object).
     * @return The result returned by provided closure.
     */
    public static Object tryGroovyClosureOn(Object closure, Object delegate) {
        try {
            Method setDelegate = closure.getClass().getMethod("setDelegate", Object.class);
            Method call = closure.getClass().getMethod("call", Object.class);
            setDelegate.invoke(closure, delegate);
            return call.invoke(closure, delegate);
        } catch (NoSuchMethodException | InvocationTargetException | IllegalAccessException e) {
            _LOG.error("Failed calling Groovy closure for loading settings!", e);
        }
        return null;
    }

    public static void tryGroovyScriptsOn( Neureka instance, Consumer<String> scriptConsumer )
    {
        if ( _settings_source == null || _setup_source == null) {
            _settings_source = instance.utility().readResource("library_settings.groovy");
            _setup_source = instance.utility().readResource("scripting_setup.groovy");
        }
        try {
            /*
            String version = GroovySystem.getVersion();
            if (Integer.parseInt(version.split("\\.")[ 0 ]) < 2) {
                throw new IllegalStateException(
                        "Wrong groovy version "+version+" found! Version 2.0.0 or greater required."
                );
            }
            */
            scriptConsumer.accept(_settings_source);
            scriptConsumer.accept(_setup_source);
        } catch (Exception e) {
            _LOG.error("Failed to load settings from Groovy script!", e);
        }

    }

}
