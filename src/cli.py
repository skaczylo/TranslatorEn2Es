import os
import subprocess
import torch
from inference import TranslatorConfig, Translator


# PALETA DE COLORES
CIAN_CLARO = "\033[96m"
AMARILLO_VIVO = "\033[1;93m" 
ROJO_SOFT = "\033[38;5;203m"
RESET = "\033[0m"
NARANJA = "\033[38;5;208m"
CIAN = "\033[36m"
AMARILLO = "\033[1;93m"  
GRIS = "\033[90m"
RESET = "\033[0m"
BOLD = "\033[1m"
ANCHO_LOGO = 76


def display_banner():
    """Imprime un banner con las letras TFG en la terminal."""
    subprocess.run('cls' if os.name == 'nt' else 'clear', shell=True)

    logo = f"""{NARANJA}{BOLD}
 ████████╗██████╗  █████╗ ██████╗ ██╗   ██╗ ██████╗████████╗ ██████╗ ██████╗ 
 ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║   ██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
    ██║   ██████╔╝███████║██║  ██║██║   ██║██║        ██║   ██║   ██║██████╔╝
    ██║   ██╔══██╗██╔══██║██║  ██║██║   ██║██║        ██║   ██║   ██║██╔══██╗
    ██║   ██║  ██║██║  ██║██████╔╝╚██████╔╝╚██████╗    ██║   ╚██████╔╝██║  ██║
    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝
                             ████████╗███████╗ ██████╗ 
                             ╚══██╔══╝██╔════╝██╔════╝ 
                                ██║   █████╗  ██║  ███╗
                                ██║   ██╔══╝  ██║   ██║
                                ██║   ██║     ╚██████╔╝
                                ╚═╝   ╚═╝      ╚═════╝ 
{RESET}"""

    texto_puro = "✽ ¡Bienvenido a la demo del Traductor Transformer! ✽"
    texto_color = f"✽ ¡Bienvenido a la demo del {AMARILLO}Traductor Transformer{RESET}{GRIS}! ✽"
    
    espacios_totales = ANCHO_LOGO - len(texto_puro) - 2
    izq = espacios_totales // 2
    der = espacios_totales - izq

    linea_sup = f"{GRIS}╭" + ("─" * (ANCHO_LOGO - 2)) + "╮"
    linea_txt = f"{GRIS}│" + (" " * izq) + f"{texto_color}" + (" " * der) + f"{GRIS}│"
    linea_inf = f"{GRIS}╰" + ("─" * (ANCHO_LOGO - 2)) + "╯"

    print(logo)
    print(linea_sup)
    print(linea_txt)
    print(linea_inf)
    
    print(f"\n{CIAN}Modelo cargado con éxito. ¡Listo para {AMARILLO}traducir{RESET}{CIAN}!{RESET}")
    print(f"{GRIS}Escribe en {AMARILLO}inglés{RESET}{GRIS} y pulsa Enter. Para salir: {AMARILLO}'salir'{RESET}{GRIS}, {AMARILLO}'exit'{RESET}{GRIS}.{RESET}\n")



def main():

    try:

        translator_cfg = TranslatorConfig()
        translator= Translator(cfg=translator_cfg)
        display_banner()

        while True:

            text_to_translate = input(f" {CIAN_CLARO}❯{RESET} Inglés:{RESET} ")

            if text_to_translate.lower() in ['exit', 'quit', 'salir']:
                print(f"\n{AMARILLO_VIVO}👋 ¡Hasta pronto!{RESET}")
                break
            if not text_to_translate.strip():
                continue

            translation = translator.translate(text_to_translate)
            print(f"    {CIAN_CLARO}Traducción:{RESET} {translation}\n")

    except Exception as e:
        print(f"\n {ROJO_SOFT}✖ Error:{RESET} {e}\n")

if __name__ == "__main__":
    main()