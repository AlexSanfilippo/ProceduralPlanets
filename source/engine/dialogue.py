"""
TODO
-[] Add support for conditions (e.g., only show certain options if a condition is met)
    -ie, player has item, quest completed, some sort of flag.
-[] Add support for actions (e.g., trigger an event when an option is selected)
       -ie, give item, start quest, etc.
-[]build tree from yaml-like file
"""



class DialogueOption:
    def __init__(self, prompt, response, menu_next=None, condition=None, action=None):
        self.prompt = prompt
        self.response = response
        self.menu_next = menu_next #a string
        self.condition = condition  #a flag to check
        self.action = action        #a flag to turn on


class DialogueMenu:
    def __init__(self):
        self.options = []
        self.flags = {}

    def add_option(self, prompt, response=None, menu_next=None, condition=None, action=None):
        self.options.append(DialogueOption(prompt, response, menu_next, condition, action))
        # add condition to flags if provided
        if condition:
            self.flags[condition] = False  # default to False

    def display(self):
        for idx, option in enumerate(self.options, start=1):
            if option.condition:
                if not self.flags.get(option.condition, False):
                    continue  # skip this option if condition not met
            print(f'{idx}) {option.prompt}')


class DialogueTree:
    """
    Handles branching dialogue options

    has a dictionary of DialogueMenus, with the key being the menu name
    """
    def __init__(self, current_menu=None, initial_text=None):
        self.menus = {}
        self.current_menu = current_menu
        self.initial_text = initial_text
          # For tracking conditions and states

    def add_menu(self, name, menu):
        self.menus[name] = menu

    def set_current_menu(self, name):
        self.current_menu = name

    def run(self):
        """Runs the dialogue tree from the current menu"""
        if self.initial_text:
            print(self.initial_text)
        while self.current_menu:
            menu = self.menus[self.current_menu]
            menu.display()
            choice = input('Please enter the number of your choice: ')
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(menu.options):
                    selected_option = menu.options[choice_idx]
                    if selected_option.action:
                        # For simplicity, we just toggle the flag for the action
                        if selected_option.action in menu.flags:
                            menu.flags[selected_option.action] = True
                    if selected_option.response:
                        print(selected_option.response)
                    if selected_option.menu_next == "EXIT":
                        self.current_menu == None
                        break
                    elif selected_option.menu_next:
                        self.current_menu = selected_option.menu_next
                else:
                    print('Invalid choice. Please try again.')
            except ValueError:
                print('Invalid input. Please enter a number.')

    def load_dialogue_file(self, filepath):
        """
        Load a dialogue file and build DialogueMenu/DialogueOption objects.

        Supported keys inside an `option:` block:
          - prompt: <text>
          - response: <text>
          - next_menu: <menu_name>  (also accepts 'menu_next')
          - action: <action_flag>
          - condition: <condition_flag>

        Usage: dialogue_tree.load_dialogue_file('source/engine/knox_station_greeter.dialogue')
        """
        current_menu = None
        current_option = None

        def _finalize_option():
            nonlocal current_option, current_menu
            if current_option and current_menu:
                # ensure prompt exists (otherwise skip)
                prompt = current_option.get('prompt')
                if prompt is not None:
                    current_menu.add_option(
                        prompt=prompt,
                        response=current_option.get('response'),
                        menu_next=current_option.get('menu_next'),
                        condition=current_option.get('condition'),
                        action=current_option.get('action'),
                    )
                current_option = None

        with open(filepath, 'r', encoding='utf-8') as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue

                if line.startswith('menu:'):
                    # finalize any pending option
                    _finalize_option()
                    name = line[len('menu:'):].strip()
                    if not name:
                        continue
                    # create menu and register
                    menu = DialogueMenu()
                    self.add_menu(name, menu)
                    current_menu = menu
                    # if no current_menu set on tree, set to first encountered
                    if not self.current_menu:
                        self.current_menu = name
                    continue

                if line.startswith('option:'):
                    # finalize previous option into current menu
                    _finalize_option()
                    current_option = {'prompt': None, 'response': None, 'menu_next': None, 'condition': None,
                                      'action': None}
                    continue

                # parse key: value lines inside an option block
                if ':' in line and current_option is not None:
                    key, val = line.split(':', 1)
                    key = key.strip()
                    val = val.strip()
                    # remove surrounding quotes if present
                    if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                        val = val[1:-1]

                    if key == 'prompt':
                        current_option['prompt'] = val
                    elif key == 'response':
                        current_option['response'] = val
                    elif key in ('next_menu', 'menu_next'):
                        current_option['menu_next'] = val
                    elif key == 'action':
                        current_option['action'] = val
                    elif key == 'condition':
                        current_option['condition'] = val
                    # unknown keys are ignored

        # finalize any remaining option
        _finalize_option()

#welcome to space station Knox diaglogue
def test_dialogue_creation_programmatically():
    dialogue_tree = DialogueTree(initial_text="Welcome to Space Station Knox! How can I assist you today?")
    main_menu = DialogueMenu()
    main_menu.add_option('What is this place?', 'This is space station Knox, a hub for interstellar travelers.')
    main_menu.add_option('Who are you?', 'I am the station AI, here to assist you.', action='is_ai_curious')
    main_menu.add_option(prompt='You\'re an AI?  What\'s that like?', response='it is alright, I suppose. I get to interact with interesting beings like yourself.', menu_next="main", condition='is_ai_curious')
    main_menu.add_option('Tell me about the station.', 'Knox is equipped with various amenities including shops, restaurants, and repair docks.', menu_next="station_info")

    main_menu.add_option('Goodbye', 'Safe travels, traveler!', menu_next="EXIT")
    dialogue_tree.add_menu(name='main', menu=main_menu)
    dialogue_tree.set_current_menu('main')
    station_info_menu = DialogueMenu()
    station_info_menu.add_option('What amenities are available?', 'We have a variety of shops, restaurants, and repair docks for your convenience.')
    station_info_menu.add_option('Are there any events happening?', 'Yes, we have a weekly market and occasional live performances in the central plaza.')
    station_info_menu.add_option('Actually, I\'d like to ask you about something else', menu_next='main')
    dialogue_tree.add_menu(name='station_info', menu=station_info_menu)
    dialogue_tree.run()

def test_dialogue_loading_from_file():
    dialogue_tree = DialogueTree()
    dialogue_tree.load_dialogue_file('knox_station_greeter.dialogue')
    dialogue_tree.run()

test_dialogue_loading_from_file()