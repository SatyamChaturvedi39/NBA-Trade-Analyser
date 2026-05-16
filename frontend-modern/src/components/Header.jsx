import React, { useState } from 'react';
import { Activity, Menu, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Header = ({ onPageChange, currentPage }) => {
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const navItems = ['home', 'model-info', 'about'];

    const handleNavClick = (page) => {
        onPageChange(page);
        setIsMenuOpen(false);
    };

    return (
        <header className="glass-panel text-white p-4 md:p-6 mb-6 md:mb-8 flex items-center justify-between relative z-50">
            <div className="flex items-center gap-3 md:gap-4">
                <div className="bg-orange-500/20 p-2 md:p-3 rounded-lg border border-orange-500/50">
                    <Activity className="w-5 h-5 md:w-8 md:h-8 text-orange-400" />
                </div>
                <div>
                    <h1
                        className="text-lg md:text-3xl font-black tracking-tight bg-gradient-to-r from-orange-400 to-rose-400 bg-clip-text text-transparent cursor-pointer"
                        onClick={() => handleNavClick('home')}
                    >
                        NBA Trade Analyzer
                    </h1>
                </div>
            </div>

            {/* Desktop Nav */}
            <nav className="hidden lg:flex items-center gap-8 mr-auto ml-12">
                {navItems.map((page) => (
                    <button
                        key={page}
                        onClick={() => handleNavClick(page)}
                        className={`text-sm font-bold uppercase tracking-widest transition-all duration-300 relative group ${currentPage === page ? 'text-orange-400' : 'text-slate-400 hover:text-white'
                            }`}
                    >
                        {page.replace('-', ' ')}
                        <span className={`absolute -bottom-1 left-0 h-0.5 bg-orange-400 transition-all duration-300 ${currentPage === page ? 'w-full' : 'w-0 group-hover:w-full'
                            }`}></span>
                    </button>
                ))}
            </nav>

            <div className="flex items-center gap-4">
                {/* Mobile Menu Toggle */}
                <button
                    className="lg:hidden p-2 text-slate-300 hover:text-white"
                    onClick={() => setIsMenuOpen(!isMenuOpen)}
                >
                    {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                </button>
            </div>

            {/* Mobile Navigation Overlay */}
            <AnimatePresence>
                {isMenuOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="absolute top-full left-0 right-0 mt-2 p-4 bg-slate-900/95 backdrop-blur-xl border border-slate-800 rounded-2xl shadow-2xl lg:hidden flex flex-col gap-4 z-50"
                    >
                        {navItems.map((page) => (
                            <button
                                key={page}
                                onClick={() => handleNavClick(page)}
                                className={`text-left text-sm font-black uppercase tracking-[0.2em] px-4 py-3 rounded-xl transition-all ${currentPage === page
                                    ? 'bg-orange-500/20 text-orange-400'
                                    : 'text-slate-400 active:bg-slate-800'
                                    }`}
                            >
                                {page.replace('-', ' ')}
                            </button>
                        ))}
                        <div className="flex sm:hidden items-center gap-2 px-4 py-3 border-t border-slate-800 mt-2">
                            <div className="w-2 h-2 rounded-full bg-emerald-400"></div>
                            <span className="text-[10px] font-bold text-slate-500 uppercase">Models Online</span>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </header>
    );
};

export default Header;
