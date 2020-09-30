import wx
from text2code import text2code as t2c

from wx.lib.agw.floatspin import FloatSpin
from wx.lib.intctrl import IntCtrl

from .. import Builder, NoParent, parser


class WXBuilder(Builder):
    """Add input_types and input_convertions attributes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_types = {
            None: wx.TextCtrl,
            'text': wx.TextCtrl,
            'checkbox': wx.CheckBox,
            'button': wx.Button,
            'int': IntCtrl,
            'float': FloatSpin
        }
        self.input_convertions = {
            'int': int,
            'float': float
        }
        self.event_globals = {
            'builder': self
        }

    @parser('frame', valid_parent=NoParent)
    def get_frame(self, parent, text, title='Untitled Frame'):
        """Return a wx.Frame instance."""
        f = wx.Frame(None, name='', title=title)
        f.panel = wx.Panel(f)
        self.event_globals['panel'] = f.panel
        self.event_globals['frame'] = f
        return f

    @parser('sizer', valid_parent=(wx.Frame, wx.BoxSizer))
    def get_sizer(
        self, parent, text, name=None, proportion='1', orient='vertical'
    ):
        """Get a sizer to add controls to."""
        orient = orient.upper()
        o = getattr(wx, orient)
        if isinstance(parent, wx.Frame):
            f = parent
        elif isinstance(parent, wx.BoxSizer):
            f = parent.frame
        else:
            raise RuntimeError(
                'Parent must be a frame or a sizer. Got: %r' % parent
            )
        s = wx.BoxSizer(o)
        s.frame = f
        if name is not None:
            setattr(f, name, s)
        yield s
        if isinstance(parent, wx.Frame):
            parent.panel.SetSizerAndFit(s)
        else:
            parent.Add(s, int(proportion), wx.GROW)

    @parser('input', valid_parent=(wx.Frame, wx.BoxSizer))
    def get_control(
        self, parent, text, proportion=1, name=None, type=None, style=None,
        label=None
    ):
        """Get a button."""
        p, s = self.get_panel_sizer(parent)
        if type not in self.input_types:
            valid_types = ', '.join(str(t) for t in self.input_types.keys())
            raise RuntimeError(
                'Invalid type: %r.\nValid types: %s' % (type, valid_types)
            )
        cls = self.input_types[type]
        kwargs = {'name': ''}
        if label is not None:
            kwargs['label'] = label
        if style is not None:
            style_int = 0
            for style_name in style.split(' '):
                style_value = getattr(wx, style_name.upper())
                style_int |= style_value
            kwargs['style'] = style_int
        c = cls(p, **kwargs)
        if name is not None:
            setattr(p.GetParent(), name, c)
        if text is not None:
            text = text.strip()
            if text:
                if type in self.input_convertions:
                    text = self.input_convertions[type](text)
                c.SetValue(text)
        s.Add(c, int(proportion), wx.GROW)
        return c

    @parser('label', valid_parent=(wx.Frame, wx.BoxSizer))
    def get_label(
        self, parent, text, name=None, proportion='0'
    ):
        """Create a label."""
        p, s = self.get_panel_sizer(parent)
        label = wx.StaticText(p, label=text.strip())
        s.Add(label, int(proportion), wx.GROW)
        if name is not None:
            setattr(p.GetParent(), name, label)
        return label

    @parser('event')
    def get_event(self, parent, text, type=None):
        """Create an event using text2code."""
        if type is None:
            raise RuntimeError('You must provide a type.')
        event_name = 'evt_' + type
        event_name = event_name.upper()
        event_type = getattr(wx, event_name)
        stuff = t2c(text, __name__, **self.event_globals)
        if 'event' not in stuff:
            raise RuntimeError(
                'No function named "event" found in %r.' % stuff
            )
        f = stuff['event']
        parent.Bind(event_type, f)
        return parent  # Don't want anything untoward happening.

    @parser('menubar', valid_parent=wx.Frame)
    def get_menubar(self, parent, text):
        """Create a menubar."""
        if parent.GetMenuBar() is not None:
            raise RuntimeError('That frame already has a menubar.')
        mb = wx.MenuBar()
        parent.SetMenuBar(mb)
        return mb

    @parser('menu', valid_parent=(wx.MenuBar, wx.Menu))
    def get_menu(self, parent, text, title=None, name=None):
        """Add a menu to the main menubar, or a submenu to an existing menu."""
        if isinstance(parent, wx.MenuBar):
            f = parent.GetTopLevelParent()
        else:
            f = parent.GetWindow()
        if title is None:
            raise RuntimeError('Menus must have a title.')
        m = wx.Menu(title)
        if name is not None:
            setattr(f, name, m)
        if isinstance(parent, wx.MenuBar):
            parent.Append(m, title)
        else:
            parent.AppendSubMenu(m, title)
        return m

    @parser('menuitem', valid_parent=wx.Menu)
    def get_menuitem(
        self, parent, text, name=None, id='any', hotkey=None, help=''
    ):
        """Adds a menu item to a menu. Must be a child tag of menu."""
        if text is None:
            raise RuntimeError(
                'This tag must contain text to be used  as the title for the '
                'menu item.'
            )
        text = text.strip()
        if hotkey is not None:
            text += f'\t{hotkey}'
        id = f'ID_{id.upper()}'
        id = getattr(wx, id)
        i = parent.Append(id, text, help)
        return i

    @parser('menuseparator', valid_parent=wx.Menu)
    def get_menu_separator(self, parent, text):
        """Add a menu separator to a menu."""
        return parent.AppendSeparator()

    @parser('menuaction', valid_parent=wx.MenuItem)
    def get_menu_action(self, parent, text):
        """Perform an action when parent is clicked."""
        d = t2c(text, __name__, **self.event_globals)
        if 'action' not in d:
            raise RuntimeError(
                'No function named "action" could be found in %r.' % d
            )
        f = parent.GetMenu().GetWindow()
        f.Bind(wx.EVT_MENU, d['action'], parent)

    def get_panel_sizer(self, parent):
        """Returns a tuple containing (panel, sizer), or raises
        AssertionError."""
        if isinstance(parent, wx.Frame):
            return (parent.panel, None)
        elif isinstance(parent, wx.BoxSizer):
            return (parent.frame.panel, parent)
        else:
            raise AssertionError(
                'Parent is neither a frame or a box sizer: %r.' % parent
            )
